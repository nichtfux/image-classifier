-- Dev check (run in Studio's command bar or via MCP): Luau rasterizer + forward pass
-- against the Python reference in TestVectors, plus a timing benchmark.
-- Expects ServerScriptService.DrawRecognizer and ServerStorage.DrawTestVectors.
local R = require(game.ServerScriptService.DrawRecognizer:Clone())
local V = require(game.ServerStorage.DrawTestVectors)
local SIZE = R.size or 28
local PADDED = SIZE + 2
assert(#V >= #R.labels, "expected references covering every category")
local covered = {}

local report = {}
local worstImg, worstLogit, agree = 0, 0, 0
local img1
for n, s in V do
	assert((s.size or 28) == SIZE, "reference/model resolution mismatch")
	assert(s.label >= 0 and s.label < #R.labels, "invalid reference label")
	covered[s.label + 1] = true
	assert(#s.logits == #R.labels, "reference/model label count mismatch")
	local strokes = {}
	for _, flat in s.strokes do
		local st = {}
		for i = 1, #flat, 2 do
			table.insert(st, Vector2.new(flat[i], flat[i + 1]))
		end
		table.insert(strokes, st)
	end
	local img = R.rasterize(strokes, s.colours)
	img1 = img1 or img
	for channel = 1, R.inputChannels or 1 do
		local sum, mom = 0, 0
		for y = 0, SIZE - 1 do
			for x = 0, SIZE - 1 do
				local v = img[(channel - 1) * PADDED * PADDED + (y + 1) * PADDED + x + 2]
				sum += v
				mom += v * (y * SIZE + x)
			end
		end
		local es = if s.imageSums then s.imageSums[channel] else s.imageSum
		local em = if s.imageMoments then s.imageMoments[channel] else s.imageMoment
		worstImg = math.max(worstImg, math.abs(sum - es), math.abs(mom - em) / (SIZE * SIZE))
	end
	local z = R.logits(img)
	local bestL, bestP = 1, 1
	for i = 1, #z do
		worstLogit = math.max(worstLogit, math.abs(z[i] - s.logits[i]))
		if z[i] > z[bestL] then bestL = i end
		if s.logits[i] > s.logits[bestP] then bestP = i end
	end
	if bestL == bestP then agree += 1 end
	table.insert(report, string.format("#%d true=%s luau=%s python=%s", n, R.labels[s.label + 1], R.labels[bestL], R.labels[bestP]))
	if n % 8 == 0 then task.wait() end
end
for i = 1, #R.labels do assert(covered[i], "missing category reference") end
assert(agree == #V, "Luau/Python top-1 mismatch")
assert(worstImg < 1e-8, "Luau/Python rasterizer mismatch")
-- The network math runs in float32 (Vector3), the Python reference in float64.
assert(worstLogit < 1e-3, "Luau/Python logits mismatch")

local times = {}
for i = 1, 41 do
	local t0 = os.clock()
	R.forward(img1)
	times[i] = (os.clock() - t0) * 1000
	if i % 8 == 0 then task.wait() end
end
table.sort(times)
local rt = {}
for i = 1, 21 do
	local t0 = os.clock()
	R.recognize({ { Vector2.new(0, 0), Vector2.new(300, 40), Vector2.new(150, 300), Vector2.new(0, 0) } }, 3)
	rt[i] = (os.clock() - t0) * 1000
	if i % 8 == 0 then task.wait() end
end
table.sort(rt)
table.insert(report, string.format(
	"samples=%d top1-agree=%d maxImageDiff=%.3g maxLogitDiff=%.3g | forward median %.2f ms (min %.2f) | recognize median %.2f ms",
	#V, agree, worstImg, worstLogit, times[21], times[1], rt[11]))
return table.concat(report, "\n")
