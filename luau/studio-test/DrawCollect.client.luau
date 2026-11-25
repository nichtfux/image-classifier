--!strict
-- DrawCollect: Studio-only prompted-word collection on top of DrawTest (2026-11-10:
-- "lets make it with prompted words"). A word to draw shows above the canvas; Rough, Good
-- or Detailed saves the drawing with that word and the author's own detail rating (for tuning the
-- detail score), Skip picks a new word. Every fifth saved drawing is kept for testing and
-- never trained on.
--
-- Drawings stay on this Studio client: no remotes, no HTTP, nothing replicated. Each save
-- is also printed to Studio's Output as a safety net. Claude exports them to
-- data/collected-drawings/ through the CollectedDrawings BindableFunction before play stops.
-- Keys: S skip (DrawTest keeps its own keys; 1-3 are colours there, so ratings are clicks).
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
local UserInputService = game:GetService("UserInputService")
local TweenService = game:GetService("TweenService")

if not RunService:IsStudio() then
	return
end

local Words = require(script:WaitForChild("Words")) :: any

local C = {
	panel = Color3.fromRGB(34, 38, 52),
	text = Color3.fromRGB(236, 238, 245),
	dim = Color3.fromRGB(150, 156, 176),
	faint = Color3.fromRGB(90, 96, 118),
	accent = Color3.fromRGB(255, 122, 47),
	bar = Color3.fromRGB(64, 70, 92),
	good = Color3.fromRGB(60, 170, 90),
}
local BOLD = Font.new("rbxasset://fonts/families/BuilderSans.json", Enum.FontWeight.Bold)
local REG = Font.new("rbxasset://fonts/families/BuilderSans.json", Enum.FontWeight.Medium)
local P_CUSTOM = 0.25 -- custom things have no real drawings at all, so ask for them more often
local TEST_EVERY = 5
local PAD, SIDE = 16, 270

local gui = Players.LocalPlayer:WaitForChild("PlayerGui"):WaitForChild("DrawTest") :: ScreenGui
local root = gui:WaitForChild("Root") :: Frame
local canvas = root:WaitForChild("Canvas") :: Frame
local ink = canvas:WaitForChild("Ink") :: Frame
local snapshotFn = gui:WaitForChild("LocalSnapshot") :: BindableFunction
local CANVAS = canvas.Size.X.Offset

local function make(class: string, props: { [string]: any }, children: { Instance }?): any
	local o = Instance.new(class)
	for k, v in props do
		(o :: any)[k] = v
	end
	for _, c in children or {} do
		c.Parent = o
	end
	return o
end
local function corner(r: number)
	return make("UICorner", { CornerRadius = UDim.new(0, r) })
end

-- header: shorten DrawTest's title so the word has room on the right
for _, c in root:GetChildren() do
	if c:IsA("TextLabel") and string.find(c.Text, "DRAW TO KILL", 1, true) then
		c.Text = "DRAW TO KILL  ·  collect"
	end
end
local prompt = make("TextLabel", {
	Name = "CollectPrompt",
	BackgroundTransparency = 1,
	RichText = true,
	FontFace = BOLD,
	TextSize = 22,
	TextColor3 = C.accent,
	TextXAlignment = Enum.TextXAlignment.Right,
	TextTruncate = Enum.TextTruncate.AtEnd,
	Position = UDim2.fromOffset(PAD + 200, 9),
	Size = UDim2.fromOffset(CANVAS - 200, 28),
	Parent = root,
})

-- the canvas hint names the word too (DrawTest shows it while the canvas is empty)
local hint: TextLabel? = nil
local hintKeys = ""
for _, c in canvas:GetChildren() do
	if c:IsA("TextLabel") then
		hint = c
		local _, e = string.find(c.Text, "\n\n", 1, true)
		hintKeys = if e then string.sub(c.Text, e + 1) else ""
	end
end

-- counter: top of the free space right of the network preview (attribution sits at its bottom)
local PREVIEW_Y = 26 + 3 * 62 + 46 + 18 -- DrawTest's preview frame, in side-panel coordinates
local counter = make("TextLabel", {
	Name = "CollectCounter",
	BackgroundTransparency = 1,
	RichText = true,
	FontFace = REG,
	TextSize = 13,
	LineHeight = 1.05,
	TextColor3 = C.dim,
	TextXAlignment = Enum.TextXAlignment.Left,
	TextYAlignment = Enum.TextYAlignment.Top,
	Position = UDim2.fromOffset(PAD + CANVAS + PAD + 122, PAD + PREVIEW_Y),
	Size = UDim2.fromOffset(SIDE - 122, 34),
	Parent = root,
})

-- row in the free strip between the network preview and the Pen / Eraser row:
-- Skip, then three save buttons that also rate the drawing's detail (2026-11-10:
-- better drawings should get a better effect; these ratings tune that score)
local row = make("Frame", {
	Name = "CollectRow",
	BackgroundTransparency = 1,
	Position = UDim2.fromOffset(PAD + CANVAS + PAD, PAD + 394),
	Size = UDim2.fromOffset(SIDE, 40),
	Parent = root,
})
local function button(name: string, text: string, x: number, w: number): TextButton
	return make("TextButton", {
		Name = name,
		Text = text,
		FontFace = BOLD,
		TextSize = 15,
		AutoButtonColor = false,
		TextColor3 = C.text,
		BackgroundColor3 = C.panel,
		Position = UDim2.fromOffset(x, 0),
		Size = UDim2.fromOffset(w, 40),
		Parent = row,
	}, { corner(8) })
end
local skipBtn = button("Skip", "Skip", 0, 56)
local RATINGS = { "Rough", "Good", "Detailed" }
local rateBtns: { TextButton } = {
	button("Rate1", "Rough", 60, 66),
	button("Rate2", "Good", 130, 66),
	button("Rate3", "Detailed", 200, 70),
}
-- three shades of the accent: the more detail, the stronger the button
local RATE_BG = { Color3.fromRGB(150, 84, 50), Color3.fromRGB(205, 103, 48), C.accent }

-- state ----------------------------------------------------------------------------
local session = os.date("%Y%m%d-%H%M%S")
local saved: { any } = {}
local recent: { string } = {}
local word, custom = "", false
local flashing = false
local needsClear = false -- saved, but DrawTest could not be cleared from here: wait for C

local function pickWord()
	local list = if math.random() < P_CUSTOM then Words.custom else Words.real
	local w
	for _ = 1, 20 do
		w = list[math.random(#list)]
		if not table.find(recent, w) then
			break
		end
	end
	word, custom = w, list == Words.custom
	table.insert(recent, w)
	if #recent > 40 then
		table.remove(recent, 1)
	end
	prompt.Text = string.format('<font color="#969CB0" size="16">draw</font>  %s', w)
	if hint then
		hint.Text = string.format("Draw: %s\n\noutline, filled or coloured – any way you like\nthen save as Rough, Good or Detailed\n\n%s\nS skip", w, hintKeys)
	end
end

local function hasInk(): boolean
	return #ink:GetChildren() > 0
end

local function refresh()
	local test = 0
	for _, r in saved do
		if r.split == "test" then
			test += 1
		end
	end
	counter.Text = string.format('<font color="#ECEEF5"><b>%d</b> saved</font>\n%d for testing', #saved, test)
	if needsClear and not hasInk() then
		needsClear = false
	end
	if flashing then
		return
	end
	local on = hasInk() and not needsClear
	for i, b in rateBtns do
		b.Text = if needsClear and i == 2 then "Press C" else RATINGS[i]
		b.BackgroundColor3 = if on then RATE_BG[i] else C.panel
		b.TextColor3 = if on then Color3.new(1, 1, 1) else C.faint
	end
end

local function clearCanvas(): boolean
	local diag = (_G :: any).DrawTestDiagnostics
	if diag and diag.clear then
		diag.clear()
		return true
	end
	return false
end

local function r1(x: number): number
	return math.round(x * 10) / 10
end

local function save(rating: number)
	if flashing or needsClear or not hasInk() then
		return
	end
	local ok, snap = pcall(function()
		return snapshotFn:Invoke(word)
	end)
	if not ok or type(snap) ~= "table" or #snap.strokes == 0 then
		warn("[DrawCollect] snapshot failed: " .. tostring(snap))
		return
	end
	local n = #saved + 1
	local strokes = {}
	for _, s in snap.strokes do
		local pts = table.create(#s.points)
		for i, p in s.points do
			pts[i] = { r1(p[1]), r1(p[2]) }
		end
		table.insert(strokes, { colour = s.colour, width = s.width, points = pts })
	end
	local top = {}
	for _, t in (if type(snap.top) == "table" then snap.top else {}) :: { any } do
		if type(t) == "table" then
			table.insert(top, { label = t.label, prob = t.prob })
		end
	end
	local rec = {
		id = string.format("%s-%03d", session, n),
		word = word,
		custom = custom,
		split = if n % TEST_EVERY == 0 then "test" else "train",
		detail = rating, -- 1 rough, 2 good, 3 detailed (The author's own rating)
		savedAt = os.time(),
		canvasSize = snap.canvasSize or CANVAS,
		model = snap.model,
		top = top,
		strokes = strokes,
	}
	table.insert(saved, rec)
	local okJson, json = pcall(HttpService.JSONEncode, HttpService, rec)
	if okJson then
		print("[DrawCollect] " .. json)
	end

	flashing = true
	needsClear = not clearCanvas()
	local b = rateBtns[rating]
	b.Text = "Saved ✓"
	b.BackgroundColor3 = C.good
	b.TextColor3 = Color3.new(1, 1, 1)
	pickWord()
	refresh()
	task.delay(0.7, function()
		flashing = false
		refresh()
	end)
end

local function skip()
	if flashing or needsClear then
		return
	end
	-- a new word must start on an empty canvas, or the old drawing gets the new word
	if hasInk() and not clearCanvas() then
		skipBtn.Text = "Press C"
		task.delay(0.9, function()
			skipBtn.Text = "Skip"
		end)
		return
	end
	pickWord()
	TweenService:Create(prompt, TweenInfo.new(0.12), { TextTransparency = 0.6 }):Play()
	task.delay(0.12, function()
		TweenService:Create(prompt, TweenInfo.new(0.18), { TextTransparency = 0 }):Play()
	end)
end

for i, b in rateBtns do
	b.Activated:Connect(function()
		save(i)
	end)
end
skipBtn.Activated:Connect(skip)
UserInputService.InputBegan:Connect(function(input, processed)
	if not processed and input.KeyCode == Enum.KeyCode.S then
		skip()
	end
end)
ink.ChildAdded:Connect(refresh)
ink.ChildRemoved:Connect(refresh)

-- export for Claude (Studio MCP): CollectedDrawings:Invoke(from?, to?) -> JSON array
local export = Instance.new("BindableFunction")
export.Name = "CollectedDrawings"
export.OnInvoke = function(from: number?, to: number?)
	local out = {}
	for i = from or 1, math.min(to or #saved, #saved) do
		table.insert(out, saved[i])
	end
	return HttpService:JSONEncode({ session = session, total = #saved, drawings = out })
end
export.Parent = gui

pickWord()
refresh()
