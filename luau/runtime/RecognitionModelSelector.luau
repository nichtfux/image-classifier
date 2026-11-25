-- Studio-only candidate selection. The canonical server recognizer is fallback.
-- In Edit, set ServerStorage.NNTestModel to a private model folder's name.
local ServerStorage = game:GetService("ServerStorage")
local RunService = game:GetService("RunService")
local fallback = script.Parent.DrawRecognizer
if not RunService:IsStudio() then return fallback end
local selected = ServerStorage:GetAttribute("NNTestModel") or "colour299_40"
local models = ServerStorage:FindFirstChild("RecognitionTestModels")
local model = models and models:FindFirstChild(selected)
assert(model, "NN test model missing: " .. tostring(selected))
if selected ~= "colour299_40" then
	assert(model:GetAttribute("Ready") == true, "NN test model is not ready: " .. tostring(selected))
end
local recognizer = if selected == "colour299_40" then fallback else model:FindFirstChild("DrawRecognizer")
assert(recognizer and recognizer:IsA("ModuleScript"), "NN test recognizer missing")
-- Only geometry/palette metadata replicates. A unique name avoids preview races
-- when selecting a different resolution before Play; no weights leave the server.
local info = assert(model:FindFirstChild("DrawModelInfo"), "NN test preview metadata missing")
local publicInfo = info:Clone()
publicInfo.Name = "DrawPreviewInfo_" .. selected
local replicated = game:GetService("ReplicatedStorage")
local previous = replicated:FindFirstChild(publicInfo.Name)
if previous then previous:Destroy() end
publicInfo.Parent = replicated
replicated:SetAttribute("DrawRecognitionPreviewInfo", publicInfo.Name)
return recognizer
