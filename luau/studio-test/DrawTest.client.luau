-- DrawTest: Studio test UI for the drawing recogniser. Draw with mouse or finger;
-- the live top-3 guesses, confidence and inference time show on the right.
-- Tools: 10 ink colours, pen and eraser, three sizes, undo / redo / clear.
-- Keys: 1-0 colour, B pen, E eraser, [ ] size, Z undo, Y (or Shift+Z) redo, C clear.
-- Strokes are sent privately to the recognition server only, never other players.
local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local UserInputService = game:GetService("UserInputService")
if not RunService:IsStudio() then return end -- test feedback is never a game submission

local Recognizer = require(ReplicatedStorage:WaitForChild("DrawRecognitionClient"))
local Palette = require(ReplicatedStorage:WaitForChild("DrawPalette"))

local CANVAS = 440 -- canvas size in layout pixels (UIScale shrinks it on small screens)
local MIN_STEP = 3 -- min distance between stored points
local LIVE_EVERY = 0.35 -- quiet time before a server guess; transport caps at ~1/s
local SIZES = { 4, 9, 16 } -- pen thickness on screen (the network always sees thin lines)
local ERASER = { 8, 14, 24 } -- eraser radius per size
local MAX_HISTORY = 200

local C = {
	bg = Color3.fromRGB(24, 27, 38),
	panel = Color3.fromRGB(34, 38, 52),
	paper = Color3.fromRGB(246, 241, 230),
	text = Color3.fromRGB(236, 238, 245),
	dim = Color3.fromRGB(150, 156, 176),
	faint = Color3.fromRGB(90, 96, 118),
	accent = Color3.fromRGB(255, 122, 47),
	bar = Color3.fromRGB(64, 70, 92),
	edge = Color3.fromRGB(60, 66, 90),
}
local BOLD = Font.new("rbxasset://fonts/families/BuilderSans.json", Enum.FontWeight.Bold)
local REG = Font.new("rbxasset://fonts/families/BuilderSans.json", Enum.FontWeight.Medium)

local function make(class, props, children)
	local o = Instance.new(class)
	for k, v in props do
		o[k] = v
	end
	for _, c in children or {} do
		c.Parent = o
	end
	return o
end
local function corner(r)
	return make("UICorner", { CornerRadius = UDim.new(0, r) })
end
local function round()
	return make("UICorner", { CornerRadius = UDim.new(1, 0) })
end
local function label(props)
	props.BackgroundTransparency = 1
	props.TextColor3 = props.TextColor3 or C.text
	props.FontFace = props.FontFace or REG
	props.TextXAlignment = props.TextXAlignment or Enum.TextXAlignment.Left
	return make("TextLabel", props)
end

-- layout -------------------------------------------------------------------------
local gui = make("ScreenGui", {
	Name = "DrawTest",
	ResetOnSpawn = false,
	IgnoreGuiInset = false,
	ZIndexBehavior = Enum.ZIndexBehavior.Sibling,
})
local SIDE = 270
local PAD = 16
local BAR = 56 -- colour bar under the canvas
local W, H = PAD + CANVAS + PAD + SIDE + PAD, PAD + 34 + CANVAS + 10 + BAR + PAD
local root = make("Frame", {
	Name = "Root",
	AnchorPoint = Vector2.new(0.5, 0.5),
	Position = UDim2.fromScale(0.5, 0.5),
	Size = UDim2.fromOffset(W, H),
	BackgroundColor3 = C.bg,
	Parent = gui,
}, { corner(14), make("UIStroke", { Color = C.edge, Thickness = 1 }) })
local scale = make("UIScale", { Parent = root })

label({
	Text = "DRAW TO KILL  ·  recogniser test",
	FontFace = BOLD,
	TextSize = 20,
	Position = UDim2.fromOffset(PAD, 10),
	Size = UDim2.fromOffset(CANVAS, 26),
	Parent = root,
})

local canvas = make("Frame", {
	Name = "Canvas",
	Position = UDim2.fromOffset(PAD, PAD + 34),
	Size = UDim2.fromOffset(CANVAS, CANVAS),
	BackgroundColor3 = C.paper,
	Active = true,
	ClipsDescendants = true,
	Parent = root,
}, { corner(10) })
local hint = label({
	Text = "Draw something here\n(anvil, sword, shark, tornado, helicopter …)\n\n1–0 colour  ·  B pen  ·  E eraser  ·  [ ] size\nZ undo  ·  Y redo  ·  C clear",
	TextColor3 = Color3.fromRGB(170, 160, 140),
	TextXAlignment = Enum.TextXAlignment.Center,
	TextSize = 18,
	Size = UDim2.fromScale(1, 1),
	Parent = canvas,
})
local inkLayer = make("Frame", { Name = "Ink", BackgroundTransparency = 1, Size = UDim2.fromScale(1, 1), Parent = canvas })
local eraserRing = make("Frame", {
	Name = "EraserRing",
	AnchorPoint = Vector2.new(0.5, 0.5),
	BackgroundColor3 = Color3.new(1, 1, 1),
	BackgroundTransparency = 0.55,
	Visible = false,
	ZIndex = 5,
	Parent = canvas,
}, { round(), make("UIStroke", { Color = Color3.fromRGB(90, 84, 72), Thickness = 1.5 }) })

-- colour bar
local bar = make("Frame", {
	Name = "Colours",
	Position = UDim2.fromOffset(PAD, PAD + 34 + CANVAS + 10),
	Size = UDim2.fromOffset(CANVAS, BAR),
	BackgroundColor3 = C.panel,
	Parent = root,
}, { corner(10) })
local SW = 34
local swatches = {}
do
	local gap = (CANVAS - 20 - #Palette * SW) / (#Palette - 1)
	for i, ink in Palette do
		local b = make("TextButton", {
			Name = ink.name,
			Text = "",
			AutoButtonColor = false,
			AnchorPoint = Vector2.new(0.5, 0.5),
			Position = UDim2.fromOffset(10 + (i - 1) * (SW + gap) + SW / 2, BAR / 2),
			Size = UDim2.fromOffset(SW, SW),
			BackgroundColor3 = ink.color,
			Parent = bar,
		}, { round() })
		local ring = make("UIStroke", {
			Color = Color3.new(1, 1, 1),
			Thickness = 0,
			ApplyStrokeMode = Enum.ApplyStrokeMode.Border,
			Parent = b,
		})
		swatches[i] = { button = b, ring = ring }
	end
end

local side = make("Frame", {
	Position = UDim2.fromOffset(PAD + CANVAS + PAD, PAD),
	Size = UDim2.fromOffset(SIDE, H - 2 * PAD),
	BackgroundTransparency = 1,
	Parent = root,
})
label({ Text = "GUESSES", FontFace = BOLD, TextSize = 14, TextColor3 = C.dim, Size = UDim2.fromOffset(SIDE, 20), Parent = side })

local rows = {}
for i = 1, 3 do
	local y = 26 + (i - 1) * 62
	local row = make("Frame", {
		Position = UDim2.fromOffset(0, y),
		Size = UDim2.fromOffset(SIDE, 54),
		BackgroundColor3 = C.panel,
		Parent = side,
	}, { corner(8) })
	local fill = make("Frame", {
		Size = UDim2.fromScale(0, 1),
		BackgroundColor3 = if i == 1 then C.accent else C.bar,
		BackgroundTransparency = if i == 1 then 0.25 else 0,
		BorderSizePixel = 0,
		Parent = row,
	}, { corner(8) })
	local name = label({
		Text = "–",
		FontFace = BOLD,
		TextSize = if i == 1 then 24 else 19,
		Position = UDim2.fromOffset(12, 0),
		Size = UDim2.new(1, -90, 1, 0),
		TextTruncate = Enum.TextTruncate.AtEnd,
		Parent = row,
	})
	local pct = label({
		Text = "",
		FontFace = BOLD,
		TextSize = 20,
		TextXAlignment = Enum.TextXAlignment.Right,
		Position = UDim2.new(1, -82, 0, 0),
		Size = UDim2.fromOffset(70, 54),
		Parent = row,
	})
	rows[i] = { fill = fill, name = name, pct = pct }
end

local timing = label({
	Text = "",
	TextSize = 15,
	TextColor3 = C.dim,
	Position = UDim2.fromOffset(0, 26 + 3 * 62),
	Size = UDim2.fromOffset(SIDE, 40),
	TextWrapped = true,
	TextYAlignment = Enum.TextYAlignment.Top,
	Parent = side,
})

-- what the network sees: the model's own size, ink composited over paper in the
-- colour the network gets (ink*r, ink*g, ink*b planes; black ink is all zero)
local NS: number = Recognizer.size or 28
local NP: number = Recognizer.paddedSize or NS + 2
local NPS: number = Recognizer.planeSize or NP * NP
local NIN: number = Recognizer.inputChannels or 1
local PREV = 112
local previewY = 26 + 3 * 62 + 46
label({ Text = string.format("NETWORK INPUT %d×%d", NS, NS), FontFace = BOLD, TextSize = 12, TextColor3 = C.dim, Position = UDim2.fromOffset(0, previewY), Size = UDim2.fromOffset(SIDE, 16), Parent = side })
local preview = make("Frame", {
	Position = UDim2.fromOffset(0, previewY + 18),
	Size = UDim2.fromOffset(PREV, PREV),
	BackgroundColor3 = C.paper,
	Parent = side,
})
local cells = table.create(NS * NS)
do
	local cs = PREV / NS
	for y = 0, NS - 1 do
		for x = 0, NS - 1 do
			cells[y * NS + x + 1] = make("Frame", {
				Position = UDim2.fromOffset(math.floor(x * cs), math.floor(y * cs)),
				Size = UDim2.fromOffset(math.ceil(cs) + 1, math.ceil(cs) + 1),
				BackgroundColor3 = C.paper,
				BackgroundTransparency = 1,
				BorderSizePixel = 0,
				Parent = preview,
			})
		end
	end
end

label({
	Text = "Model trained on Quick, Draw! by Google (CC BY 4.0)",
	TextSize = 11,
	TextColor3 = Color3.fromRGB(110, 116, 136),
	Position = UDim2.new(0, PREV + 10, 0, previewY + 18),
	Size = UDim2.fromOffset(SIDE - PREV - 10, PREV),
	TextWrapped = true,
	TextYAlignment = Enum.TextYAlignment.Bottom,
	Parent = side,
})

-- tool row: pen, eraser, three sizes
local TOOL_Y = H - 2 * PAD - 44 - 10 - 44
local function toolButton(text, x, w)
	return make("TextButton", {
		Text = text,
		FontFace = BOLD,
		TextSize = 16,
		TextColor3 = C.text,
		AutoButtonColor = true,
		BackgroundColor3 = C.panel,
		Position = UDim2.fromOffset(x, TOOL_Y),
		Size = UDim2.fromOffset(w, 44),
		Parent = side,
	}, { corner(8) })
end
local penBtn = toolButton("Pen", 0, 60)
local eraserBtn = toolButton("Eraser", 66, 60)
local sizeBtns = {}
for i = 1, 3 do
	local b = toolButton("", 138 + (i - 1) * 44, 40)
	local d = SIZES[i] + 2
	local dot = make("Frame", {
		AnchorPoint = Vector2.new(0.5, 0.5),
		Position = UDim2.fromScale(0.5, 0.5),
		Size = UDim2.fromOffset(d, d),
		BorderSizePixel = 0,
		Parent = b,
	}, { round() })
	local ring = make("UIStroke", { Color = C.accent, Thickness = 0, ApplyStrokeMode = Enum.ApplyStrokeMode.Border, Parent = b })
	sizeBtns[i] = { button = b, dot = dot, ring = ring }
end

-- action row: undo, redo, clear
local BW = (SIDE - 16) / 3
local function actionButton(text, i)
	return make("TextButton", {
		Text = text,
		FontFace = BOLD,
		TextSize = 18,
		TextColor3 = C.text,
		AutoButtonColor = true,
		BackgroundColor3 = C.panel,
		Position = UDim2.new(0, (i - 1) * (BW + 8), 1, -44),
		Size = UDim2.fromOffset(BW, 44),
		Parent = side,
	}, { corner(8) })
end
local undoBtn = actionButton("Undo", 1)
local redoBtn = actionButton("Redo", 2)
local clearBtn = actionButton("Clear", 3)

local function fit()
	local cam = workspace.CurrentCamera
	local vp = cam and cam.ViewportSize or Vector2.new(1280, 720)
	scale.Scale = math.min(1, (vp.X - 16) / W, (vp.Y - 60) / H)
end

-- state ----------------------------------------------------------------------------
-- A stroke is { pts = {Vector2}, color = palette index, size = size index }. Finished
-- strokes are never changed: the eraser replaces them with new pieces, so history
-- snapshots can share them.
type Stroke = { pts: { Vector2 }, color: number, size: number }
local strokes: { Stroke } = {}
local inkFrames: { [Stroke]: { Frame } } = {}
local history: { { Stroke } } = { {} }
local histAt = 1

local tool = "pen"
local colorIndex = 1
local sizeIndex = 2

local activeInput: InputObject? = nil
local activeStroke: Stroke? = nil
local lastErase: Vector2? = nil
local eraseChanged = false
local dirty = false
local revision = 0
local currentJob = nil
local jobRevision = 0
local lastStats, lastTop = nil, nil
local resultRevision = -1
local SLICE_MS = 1.25
local lastEditAt = 0

local function invalidateGuess()
	lastEditAt = os.clock()
	revision += 1
	dirty = true
	if currentJob then
		currentJob:cancel()
		currentJob = nil
	end
end

local function dot(p: Vector2, s: Stroke)
	local w = SIZES[s.size]
	return make("Frame", {
		AnchorPoint = Vector2.new(0.5, 0.5),
		Position = UDim2.fromOffset(p.X, p.Y),
		Size = UDim2.fromOffset(w, w),
		BackgroundColor3 = Palette[s.color].color,
		BorderSizePixel = 0,
		Parent = inkLayer,
	}, { round() })
end
local function segment(a: Vector2, b: Vector2, s: Stroke)
	local d = b - a
	local mid = (a + b) / 2
	return make("Frame", {
		AnchorPoint = Vector2.new(0.5, 0.5),
		Position = UDim2.fromOffset(mid.X, mid.Y),
		Size = UDim2.fromOffset(d.Magnitude, SIZES[s.size]),
		Rotation = math.deg(math.atan2(d.Y, d.X)),
		BackgroundColor3 = Palette[s.color].color,
		BorderSizePixel = 0,
		Parent = inkLayer,
	})
end

local function buildInk(s: Stroke)
	local list = {}
	for i, p in s.pts do
		if i > 1 then
			table.insert(list, segment(s.pts[i - 1], p, s))
		end
		table.insert(list, dot(p, s))
	end
	inkFrames[s] = list
end
local function dropInk(s: Stroke)
	for _, f in inkFrames[s] or {} do
		f:Destroy()
	end
	inkFrames[s] = nil
end

-- show a list of strokes: keep frames of strokes that stay, build the new ones
local function setStrokes(list: { Stroke })
	local keep = {}
	for _, s in list do
		keep[s] = true
	end
	for s in inkFrames do
		if not keep[s] then
			dropInk(s)
		end
	end
	for _, s in list do
		if not inkFrames[s] then
			buildInk(s)
		end
	end
	strokes = list
	hint.Visible = #strokes == 0
	invalidateGuess()
end

local function refreshButtons()
	local canUndo, canRedo = histAt > 1, histAt < #history
	undoBtn.TextColor3 = if canUndo then C.text else C.faint
	redoBtn.TextColor3 = if canRedo then C.text else C.faint
	clearBtn.TextColor3 = if #strokes > 0 then C.text else C.faint
end

local function pushHistory()
	for i = #history, histAt + 1, -1 do
		history[i] = nil
	end
	table.insert(history, table.clone(strokes))
	if #history > MAX_HISTORY then
		table.remove(history, 1)
	end
	histAt = #history
	refreshButtons()
end

local function refreshTools()
	local erasing = tool == "eraser"
	penBtn.BackgroundColor3 = if erasing then C.panel else C.accent
	eraserBtn.BackgroundColor3 = if erasing then C.accent else C.panel
	for i, sw in swatches do
		local on = i == colorIndex
		sw.ring.Thickness = if on then 3 else 0
		sw.button.Size = if on then UDim2.fromOffset(SW + 4, SW + 4) else UDim2.fromOffset(SW, SW)
	end
	for i, sb in sizeBtns do
		-- black ink would vanish on the dark button, so its dots show light
		sb.dot.BackgroundColor3 = if erasing then C.dim elseif colorIndex == 1 then C.text else Palette[colorIndex].color
		sb.ring.Thickness = if i == sizeIndex then 2 else 0
	end
	eraserRing.Size = UDim2.fromOffset(ERASER[sizeIndex] * 2, ERASER[sizeIndex] * 2)
	if not erasing then
		eraserRing.Visible = false
	end
end

local function setTool(t)
	if activeInput then
		return
	end
	tool = t
	refreshTools()
end
local function setColor(i)
	colorIndex = i
	setTool("pen")
end
local function setSize(i)
	sizeIndex = math.clamp(i, 1, #SIZES)
	refreshTools()
end

local function toCanvas(pos: Vector3): Vector2
	local abs, size = canvas.AbsolutePosition, canvas.AbsoluteSize
	local p = (Vector2.new(pos.X, pos.Y) - abs) * (CANVAS / size.X)
	return Vector2.new(math.clamp(p.X, 0, CANVAS), math.clamp(p.Y, 0, CANVAS))
end
local function overCanvas(pos: Vector3): boolean
	local abs, size = canvas.AbsolutePosition, canvas.AbsoluteSize
	return pos.X >= abs.X and pos.Y >= abs.Y and pos.X <= abs.X + size.X and pos.Y <= abs.Y + size.Y
end

-- recognition ----------------------------------------------------------------------
local function showResult(top, rasterMs, netMs, image, stats)
	lastStats, lastTop = stats, top
	for i = 1, 3 do
		local r, t = rows[i], top[i]
		if t then
			r.name.Text = t.label
			r.pct.Text = string.format("%d%%", math.floor(t.prob * 100 + 0.5))
			r.fill.Size = UDim2.fromScale(math.clamp(t.prob, 0.02, 1), 1)
		else
			r.name.Text = "–"
			r.pct.Text = ""
			r.fill.Size = UDim2.fromScale(0, 1)
		end
	end
	if netMs then
		if stats and stats.server then
			timing.Text = string.format("net %.1f ms · RTT %.0f ms\nserver %.1f ms · slice %.2f ms", netMs, stats.wallMs, stats.activeMs, stats.maxSliceMs)
		elseif stats then
			timing.Text = string.format("net %.1f ms · wall %.0f ms\nraster %.1f ms · slice %.2f ms", netMs, stats.wallMs, rasterMs, stats.maxSliceMs)
		else
			timing.Text = string.format("network %.1f ms  ·  raster %.1f ms", netMs, rasterMs)
		end
	else
		timing.Text = ""
	end
	-- premultiplied colour over paper: RGB + paper*(1-ink)
	local pr, pg, pb = C.paper.R, C.paper.G, C.paper.B
	for i = 1, NS * NS do
		local cell = cells[i]
		local k = (((i - 1) // NS) + 1) * NP + ((i - 1) % NS) + 2
		local v = image and image[k] or 0
		if v <= 0 then
			cell.BackgroundTransparency = 1
		else
			local r, g, b = 0, 0, 0
			if NIN == 4 then
				r, g, b = image[k + NPS], image[k + 2 * NPS], image[k + 3 * NPS]
			end
			local rest = 1 - math.min(v, 1)
			cell.BackgroundColor3 = Color3.new(
				math.clamp(r + pr * rest, 0, 1),
				math.clamp(g + pg * rest, 0, 1),
				math.clamp(b + pb * rest, 0, 1)
			)
			cell.BackgroundTransparency = 0
		end
	end
end

local function guess()
	if activeInput or currentJob or not Recognizer.canRequest() then return end
	dirty = false
	local pts, cols, widths = {}, {}, {}
	for _, s in strokes do
		if #s.pts > 0 then
			table.insert(pts, s.pts)
			table.insert(cols, s.color)
			table.insert(widths, SIZES[s.size])
		end
	end
	if #pts == 0 then
		showResult({}, nil, nil, nil)
		return
	end
	-- per-stroke colours (DrawPalette indices); models without colour ignore them
	currentJob = Recognizer.beginRecognize(pts, 3, cols, widths)
	jobRevision = revision
	timing.Text = "Recognising…"
end

-- pen ------------------------------------------------------------------------------
local function addPoint(p: Vector2, terminal: boolean?)
	local s = activeStroke :: Stroke
	local last = s.pts[#s.pts]
	if last and (p - last).Magnitude < (if terminal then 0.01 else MIN_STEP) then
		return
	end
	local list = inkFrames[s]
	if last then
		table.insert(list, segment(last, p, s))
	end
	table.insert(list, dot(p, s))
	table.insert(s.pts, p)
	invalidateGuess()
end

-- eraser ---------------------------------------------------------------------------
-- Cut the parts of a stroke inside the circle (c, r). Returns nil when the stroke is
-- untouched, else the remaining pieces (point lists, possibly none).
local function cutStroke(pts: { Vector2 }, c: Vector2, r: number): { { Vector2 } }?
	if #pts == 1 then
		return if (pts[1] - c).Magnitude <= r then {} else nil
	end
	local pieces, cur, touched = {}, nil, false
	local function emit(v: Vector2)
		if not cur then
			cur = { v }
		elseif (cur[#cur] - v).Magnitude > 0.01 then
			table.insert(cur, v)
		end
	end
	local function cut()
		if cur then
			local len = 0
			for i = 2, #cur do
				len += (cur[i] - cur[i - 1]).Magnitude
			end
			if len >= 2 then
				table.insert(pieces, cur)
			end
			cur = nil
		end
	end
	for i = 1, #pts - 1 do
		local a, b = pts[i], pts[i + 1]
		local d, f = b - a, a - c
		local A = d:Dot(d)
		local hit = false
		local t1, t2 = 0, 1
		if A < 1e-9 then
			hit = f.Magnitude <= r
		else
			local B, Cc = 2 * f:Dot(d), f:Dot(f) - r * r
			local disc = B * B - 4 * A * Cc
			if disc > 0 then
				local sq = math.sqrt(disc)
				t1, t2 = (-B - sq) / (2 * A), (-B + sq) / (2 * A)
				hit = t1 < 1 and t2 > 0
			end
		end
		if not hit then
			emit(a)
			emit(b)
		else
			touched = true
			if t1 > 0 then
				emit(a)
				emit(a + d * t1)
			end
			cut()
			if t2 < 1 then
				emit(a + d * t2)
				emit(b)
			end
		end
	end
	cut()
	return if touched then pieces else nil
end

local function eraseAt(c: Vector2)
	local r = ERASER[sizeIndex]
	local out, changed = {}, false
	for _, s in strokes do
		local pieces = cutStroke(s.pts, c, r + SIZES[s.size] / 2)
		if pieces then
			changed = true
			for _, p in pieces do
				table.insert(out, { pts = p, color = s.color, size = s.size })
			end
		else
			table.insert(out, s)
		end
	end
	if changed then
		eraseChanged = true
		setStrokes(out)
	end
end

local function eraseTo(p: Vector2)
	local from = lastErase or p
	local dist = (p - from).Magnitude
	local step = ERASER[sizeIndex] * 0.5
	local n = math.max(1, math.ceil(dist / step))
	for i = 1, n do
		eraseAt(from:Lerp(p, i / n))
	end
	lastErase = p
end

local function moveRing(pos: Vector3)
	if tool == "eraser" and overCanvas(pos) then
		local p = toCanvas(pos)
		eraserRing.Position = UDim2.fromOffset(p.X, p.Y)
		eraserRing.Visible = true
	else
		eraserRing.Visible = false
	end
end

-- actions --------------------------------------------------------------------------
local function undo()
	if activeInput or histAt <= 1 then
		return
	end
	histAt -= 1
	setStrokes(table.clone(history[histAt]))
	refreshButtons()
end
local function redo()
	if activeInput or histAt >= #history then
		return
	end
	histAt += 1
	setStrokes(table.clone(history[histAt]))
	refreshButtons()
end
local function clear()
	if activeInput or #strokes == 0 then
		return
	end
	setStrokes({})
	pushHistory()
end

-- input ----------------------------------------------------------------------------
canvas.InputBegan:Connect(function(input)
	local t = input.UserInputType
	if activeInput or (t ~= Enum.UserInputType.MouseButton1 and t ~= Enum.UserInputType.Touch) then
		return
	end
	activeInput = input
	local p = toCanvas(input.Position)
	if tool == "eraser" then
		lastErase, eraseChanged = nil, false
		moveRing(input.Position)
		eraseTo(p)
	else
		hint.Visible = false
		local s: Stroke = { pts = {}, color = colorIndex, size = sizeIndex }
		activeStroke = s
		inkFrames[s] = {}
		table.insert(strokes, s)
		addPoint(p)
	end
end)

UserInputService.InputChanged:Connect(function(input)
	if input.UserInputType == Enum.UserInputType.MouseMovement then
		moveRing(input.Position)
	end
	if not activeInput then
		return
	end
	if activeInput.UserInputType == Enum.UserInputType.Touch then
		if input ~= activeInput then
			return
		end
		moveRing(input.Position)
	elseif input.UserInputType ~= Enum.UserInputType.MouseMovement then
		return
	end
	local p = toCanvas(input.Position)
	if tool == "eraser" then
		eraseTo(p)
	else
		addPoint(p)
	end
end)

UserInputService.InputEnded:Connect(function(input)
	if not activeInput then
		return
	end
	if input == activeInput or input.UserInputType == activeInput.UserInputType and input.UserInputType == Enum.UserInputType.MouseButton1 then
		local wasTouch = activeInput.UserInputType == Enum.UserInputType.Touch
		-- InputChanged can be coalesced while the frame is busy. Preserve the
		-- release position even when it is closer than the regular point spacing.
		if activeStroke then addPoint(toCanvas(input.Position), true) end
		activeInput = nil
		if activeStroke then
			activeStroke = nil
			pushHistory()
		elseif eraseChanged then
			pushHistory()
		end
		lastErase, eraseChanged = nil, false
		if wasTouch then
			eraserRing.Visible = false
		end
		guess()
	end
end)

local DIGITS = {
	[Enum.KeyCode.One] = 1, [Enum.KeyCode.Two] = 2, [Enum.KeyCode.Three] = 3, [Enum.KeyCode.Four] = 4,
	[Enum.KeyCode.Five] = 5, [Enum.KeyCode.Six] = 6, [Enum.KeyCode.Seven] = 7, [Enum.KeyCode.Eight] = 8,
	[Enum.KeyCode.Nine] = 9, [Enum.KeyCode.Zero] = 10,
}
UserInputService.InputBegan:Connect(function(input, processed)
	if processed then
		return
	end
	local k = input.KeyCode
	local shift = UserInputService:IsKeyDown(Enum.KeyCode.LeftShift) or UserInputService:IsKeyDown(Enum.KeyCode.RightShift)
	if DIGITS[k] and Palette[DIGITS[k]] then
		setColor(DIGITS[k])
	elseif k == Enum.KeyCode.B or k == Enum.KeyCode.P then
		setTool("pen")
	elseif k == Enum.KeyCode.E then
		setTool("eraser")
	elseif k == Enum.KeyCode.LeftBracket then
		setSize(sizeIndex - 1)
	elseif k == Enum.KeyCode.RightBracket then
		setSize(sizeIndex + 1)
	elseif k == Enum.KeyCode.Z then
		if shift then
			redo()
		else
			undo()
		end
	elseif k == Enum.KeyCode.Y then
		redo()
	elseif k == Enum.KeyCode.C then
		clear()
	end
end)

for i, sw in swatches do
	sw.button.Activated:Connect(function()
		setColor(i)
	end)
end
for i, sb in sizeBtns do
	sb.button.Activated:Connect(function()
		setSize(i)
	end)
end
penBtn.Activated:Connect(function()
	setTool("pen")
end)
eraserBtn.Activated:Connect(function()
	setTool("eraser")
end)
undoBtn.Activated:Connect(undo)
redoBtn.Activated:Connect(redo)
clearBtn.Activated:Connect(clear)

local since = 0
RunService.Heartbeat:Connect(function(dt)
	since += dt
	if dirty and not activeInput and since >= LIVE_EVERY and os.clock()-lastEditAt >= LIVE_EVERY then
		since = 0
		guess()
	end
	if currentJob then
		local job = currentJob
		-- Only the weight-free preview runs here; inference is server-budgeted.
		local budget = math.clamp(SLICE_MS * dt * 60, 1, 2)
		local ok, done, result, stats = pcall(job.step, job, budget)
		if not ok then
			currentJob = nil
			showResult({}, nil, nil, nil)
			timing.Text = "Recognition failed"
			warn("[DrawTest] recognition failed: " .. tostring(done))
		elseif done then
			currentJob = nil
			if job.state == "completed" and jobRevision == revision then
				resultRevision = revision
				if result.error then
					showResult({}, nil, nil, result.image)
					timing.Text = "Recognition: " .. tostring(result.error)
				else
					showResult(result.top, result.rasterMs, result.networkMs, result.image, stats)
				end
			end
		end
	end
end)

refreshTools()
refreshButtons()
fit()
workspace.CurrentCamera:GetPropertyChangedSignal("ViewportSize"):Connect(fit)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

-- Developer inspection only. No output, remotes, HTTP, replication or automatic
-- collection. A caller must explicitly request a snapshot on this Studio client.
-- A future label/export UI is Claude's task and needs the author's approval first.
if RunService:IsStudio() then
	_G.DrawTestDiagnostics = {
		clear = clear,
		snapshot = function(intendedLabel: string?)
			local data = { version = 1, intendedLabel = intendedLabel, model = Recognizer.modelName,
				canvasSize = CANVAS, brushDisplayOnly = true, strokes = {}, top = lastTop,
				stats = lastStats, active = activeInput ~= nil, revision = revision,
				resultRevision = resultRevision, pending = currentJob ~= nil }
			for _,s in strokes do
				local pts = {}
				for _,p in s.pts do pts[#pts+1] = {p.X,p.Y} end
				data.strokes[#data.strokes+1] = {points = pts, colour = s.color, width = SIZES[s.size]}
			end
			return data
		end,
	}
	local inspect = Instance.new("BindableFunction")
	inspect.Name = "LocalSnapshot"
	inspect.OnInvoke = _G.DrawTestDiagnostics.snapshot
	inspect.Parent = gui -- client-created, Studio-only; never replicated
end
