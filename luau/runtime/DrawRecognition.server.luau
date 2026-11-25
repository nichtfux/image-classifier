-- Finished drawing submissions only in game. Frequent feedback is Studio-only.
local Players=game:GetService("Players")
local RunService=game:GetService("RunService")
local remote=game.ReplicatedStorage:WaitForChild("SubmitDrawing")
local service=require(script.Parent.RecognitionService)
remote.OnServerEvent:Connect(function(player,id,token,payload,...)
	if select("#",...)~=0 then payload=nil end -- reject extra client arguments
	service:submit(player,id,token,payload)
end)
Players.PlayerRemoving:Connect(function(player) service:remove(player) end)
RunService.Heartbeat:Connect(function() service.queue:tick() end)

-- Studio MCP require() may have its own cache. Inspect the actual running Script's
-- singleton through this server-only, Studio-only bindable, not a second require.
if RunService:IsStudio() then
	local completed=0
	service.Completed:Connect(function() completed+=1 end)
	local preview=game.ReplicatedStorage:WaitForChild("DrawRecognitionPreview")
	preview.OnServerEvent:Connect(function(player,id,payload,...)
		if select("#",...)~=0 then payload=nil end
		service.queue:request(player,id,payload,false) -- cannot create trusted game results
	end)
	local inspect=Instance.new("BindableFunction")
	inspect.Name="RecognitionInspect"
	inspect.OnInvoke=function(command,player,id)
		if command=="model" then return service.ModelName end
		if command=="metrics" then return table.clone(service.queue.metrics) end
		if command=="completed" then return completed end
		if command=="consume" then return service:consume(player,id) end
		if command=="depth" then return service.queue:depth() end
		if command=="open" then return service:openDrawing(player,{studioProbe=true}) end
		return nil
	end
	inspect.Parent=game.ServerStorage
else
	local preview=game.ReplicatedStorage:FindFirstChild("DrawRecognitionPreview")
	if preview then preview:Destroy() end -- test endpoint does not exist in live servers
end
