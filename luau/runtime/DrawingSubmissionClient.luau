-- Game path: call submit exactly once from the finished-drawing action, using a
-- ticket issued privately by the game server. No preview model, timer or streaming.
local P=require(game.ReplicatedStorage.DrawProtocol)
local remote=game.ReplicatedStorage:WaitForChild("SubmitDrawing")
local M={}
local nextId,pending=0,nil
local function deliver(entry,response)
	if pending~=entry then return end
	pending=nil
	response.roundTripMs=(os.clock()-entry.startedAt)*1000
	local ok,err=pcall(entry.callback,response)
	if not ok then warn("[DrawingSubmit] callback failed: "..tostring(err)) end
end
remote.OnClientEvent:Connect(function(id,response)
	if pending and pending.id==id and type(response)=="table" then deliver(pending,response) end
end)
function M.submit(ticket,strokes,colours,widths,callback)
	if pending then return nil,"in_flight" end
	if type(ticket)~="string" or #ticket>64 or type(callback)~="function" then return nil,"submission_arguments" end
	local payload,err=P.encode(strokes,colours,widths)
	if not payload then return nil,err end
	nextId+=1
	local entry={id=nextId,startedAt=os.clock(),callback=callback}
	pending=entry
	remote:FireServer(entry.id,ticket,payload)
	task.delay(8,function()
		if pending==entry then deliver(entry,{ok=false,error="server_timeout"}) end
	end)
	return entry.id
end
return M
