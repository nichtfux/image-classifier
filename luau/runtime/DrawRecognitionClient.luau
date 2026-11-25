-- STUDIO TEST TOOL ONLY: whole-canvas snapshots for frequent diagnostic feedback.
assert(game:GetService("RunService"):IsStudio(),"DrawRecognitionClient is Studio-only")
local ReplicatedStorage=game:GetService("ReplicatedStorage")
local Protocol=require(ReplicatedStorage.DrawProtocol)
local deadline=os.clock()+20
while not ReplicatedStorage:GetAttribute("DrawRecognitionPreviewInfo") do
	assert(os.clock()<deadline,"server preview model was not initialized")
	task.wait()
end
local previewInfo=ReplicatedStorage:WaitForChild(ReplicatedStorage:GetAttribute("DrawRecognitionPreviewInfo"),20)
assert(previewInfo and previewInfo:IsA("ModuleScript"),"selected preview metadata missing")
local Rasterizer=require(ReplicatedStorage.DrawRasterizer)(require(previewInfo))
local remote=ReplicatedStorage:WaitForChild("DrawRecognitionPreview")
local M={}
for _,key in {"size","paddedSize","planeSize","inputChannels","modelName"} do M[key]=Rasterizer[key] end
local nextId,nextSend,pending=0,0,nil
remote.OnClientEvent:Connect(function(id,response)
	if pending and id==pending.id and type(response)=="table" then
		if type(response.model)=="string" then M.modelName=response.model end
		pending.response=response
		pending.rttMs=(os.clock()-pending.sentAt)*1000
		pending.done=true
		pending=nil
	end
end)

local function expire()
	if pending and os.clock()-pending.sentAt>8 then
		pending.response={ok=false,error="server_timeout"};pending.rttMs=(os.clock()-pending.sentAt)*1000;pending.done=true
		pending=nil
	end
end
function M.canRequest()
	expire()
	return pending==nil and os.clock()>=nextSend
end

function M.beginRecognize(strokes,k,colours,widths)
	assert(M.canRequest(),"recognition transport busy")
	local payload,canonical=Protocol.encode(strokes,colours,widths)
	local job={state="pending",result=nil}
	if not payload then
		function job:step() self.state="completed";return true,{top={},error=canonical},{} end
		function job:cancel() self.state="cancelled" end
		return job
	end
	nextId+=1
	local entry={id=nextId,sentAt=os.clock(),done=false}
	pending=entry
	nextSend=entry.sentAt+1.05
	local preview=Rasterizer.beginRasterize(canonical.strokes,canonical.colours)
	local image,previewDone
	remote:FireServer(entry.id,payload)
	function job:step(budgetMs)
		if self.state=="cancelled" then return true,nil,{} end
		if self.state=="completed" then return true,self.result,self.stats end
		self.state="running"
		if not previewDone then previewDone,image=preview:step(budgetMs or 1.25) end
		expire()
		if not entry.done or not previewDone then return false end
		local response=entry.response
		local stats=response.stats or {}
		stats.server=true
		stats.serverWallMs=stats.wallMs
		stats.wallMs=entry.rttMs -- client-measured round trip, includes queue + wire
		stats.requestId=entry.id
		stats.previewMs=preview.stats.activeMs
		stats.maxPreviewSliceMs=preview.stats.maxSliceMs
		self.stats=stats
		self.result={top=response.top or {},image=image,rasterMs=stats.rasterMs or 0,
			networkMs=stats.networkMs or 0,error=if response.ok then nil else response.error}
		self.state="completed"
		return true,self.result,stats
	end
	function job:cancel()
		if self.state=="completed" then return end
		self.state="cancelled";preview:cancel()
		-- Keep the transport in-flight until its reply/deadline. Edits cannot
		-- bypass the server's one-request rule or create an unbounded queue.
	end
	return job
end

return M
