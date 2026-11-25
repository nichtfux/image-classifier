-- Server-only singleton. A future game consumes trusted results from this module,
-- never a label/probability claimed by the client.
local ReplicatedStorage=game:GetService("ReplicatedStorage")
local recognizerModule=require(script.Parent.RecognitionModelSelector)
local R=require(recognizerModule)
local createQueue=require(script.Parent.RecognitionQueue)
local protocol=require(ReplicatedStorage.DrawProtocol)
local job=require(recognizerModule.DrawRasterizer.DrawJob)
local function reply(player,id,result)
	local name=if result.mode=="preview" then "DrawRecognitionPreview" else "SubmitDrawing"
	local remote=ReplicatedStorage:FindFirstChild(name)
	if player.Parent==game.Players and remote then remote:FireClient(player,id,result) end
end
local completed=Instance.new("BindableEvent") -- private, unparented server object
local queue=createQueue(R,protocol,job,{reply=reply,completed=function(player,id) completed:Fire(player,id) end})
local service=require(script.Parent.DrawingSubmissionGate)(queue,{
	issue=function() return game:GetService("HttpService"):GenerateGUID(false) end,
	reply=reply,
})
service.Completed=completed.Event -- server gameplay consumes the trusted result here
service.ModelName=R.modelName
return service
