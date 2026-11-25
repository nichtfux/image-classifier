--!native
--!optimize 2
-- Weight-free preview; identical server/client geometry.
local newJob = require(script.DrawJob)
return function(Weights)
local M = {}
local SIZE: number = Weights.size or 28 -- older exports did not store their input resolution
local PADDED: number = SIZE + 2
M.size = SIZE
M.paddedSize = PADDED
M.modelName = Weights.name or "legacy"
local BOX = Weights.box
local RADIUS = Weights.radius
local REACH = RADIUS + 0.5
local V3 = Vector3.new
local INPUTS: number = Weights.inputChannels or Weights.layers[1].cin or 1
local PS: number = PADDED * PADDED
local TIE_EPS: number = Weights.ownerTieEpsilon or 1e-12
local paletteNet = {}
for i, rgb in Weights.paletteNet or {} do paletteNet[i] = V3(rgb[1], rgb[2], rgb[3]) end
assert(INPUTS == 1 or INPUTS == 4, "unsupported input planes")
if INPUTS == 4 then assert(#paletteNet == 10, "missing model palette") end
M.inputChannels = INPUTS
M.planeSize = PS
M.paletteNet = paletteNet

-- rasterizer -----------------------------------------------------------------
-- strokes: array of arrays of Vector2 (any screen space, y down).
-- Returns INPUTS padded planes (ink first); per-plane index is
-- (y+1)*PADDED + x + 2. Nil/missing/invalid colour indices default to black.
-- or nil if there is nothing drawn.
local function rasterize(strokes, colours: { number }?, checkpoint): { number }?
	local minx, miny, maxx, maxy = math.huge, math.huge, -math.huge, -math.huge
	local any = false
	local pointCount = 0
	for _, st in strokes do
		for _, p in st do
			any = true
			if p.X < minx then minx = p.X end
			if p.X > maxx then maxx = p.X end
			if p.Y < miny then miny = p.Y end
			if p.Y > maxy then maxy = p.Y end
			pointCount += 1
			if checkpoint and pointCount % 128 == 0 then checkpoint() end
		end
	end
	if not any then
		return nil
	end
	local size = math.max(maxx - minx, maxy - miny)
	local scale = if size > 1e-6 then BOX / size else 1.0
	local cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
	local half = SIZE / 2
	local strokeColour = {}
	local coloured = false
	if INPUTS == 4 and colours then
		for si, _ in strokes do
			local c = colours[si]
			if type(c) ~= "number" or c ~= c or c % 1 ~= 0 or c < 1 or c > #paletteNet then c = 1 end
			strokeColour[si] = c
			if c > 1 then coloured = true end
			if checkpoint and si % 128 == 0 then checkpoint() end
		end
	end

	local dmin = table.create(SIZE * SIZE, math.huge) -- squared distance per pixel
	local owner = if coloured then table.create(SIZE * SIZE, 1) else nil
	local function seg(si, ax, ay, bx, by)
		local dx, dy = bx - ax, by - ay
		local ll = dx * dx + dy * dy
		local x0 = math.max(0, math.ceil(math.min(ax, bx) - REACH - 0.5))
		local x1 = math.min(SIZE - 1, math.floor(math.max(ax, bx) + REACH - 0.5))
		local y0 = math.max(0, math.ceil(math.min(ay, by) - REACH - 0.5))
		local y1 = math.min(SIZE - 1, math.floor(math.max(ay, by) + REACH - 0.5))
		for py = y0, y1 do
			local fy = py + 0.5
			local row = py * SIZE + 1
			for px = x0, x1 do
				local fx = px + 0.5
				local t = 0.0
				if ll > 1e-12 then
					t = ((fx - ax) * dx + (fy - ay) * dy) / ll
					if t < 0 then t = 0 elseif t > 1 then t = 1 end
				end
				local ex = fx - (ax + t * dx)
				local ey = fy - (ay + t * dy)
				local d2 = ex * ex + ey * ey
				if owner and d2 <= dmin[row + px] + TIE_EPS then owner[row + px] = si end
				if d2 < dmin[row + px] then
					dmin[row + px] = d2
				end
			end
			if checkpoint and py % 8 == 7 then checkpoint() end
		end
	end
	for si, st in strokes do
		local n = #st
		if n == 0 then continue end
		local p = st[1]
		local ax, ay = (p.X - cx) * scale + half, (p.Y - cy) * scale + half
		if n == 1 then
			seg(si, ax, ay, ax, ay)
		end
		for i = 2, n do
			local q = st[i]
			local bx, by = (q.X - cx) * scale + half, (q.Y - cy) * scale + half
			seg(si, ax, ay, bx, by)
			ax, ay = bx, by
			if checkpoint and i % 16 == 0 then checkpoint() end
		end
		if checkpoint then checkpoint() end
	end

	local img = table.create(INPUTS * PS, 0)
	for y = 0, SIZE - 1 do
		for x = 0, SIZE - 1 do
			local d2 = dmin[y * SIZE + x + 1]
			if d2 < REACH * REACH then
				local v = REACH - math.sqrt(d2)
				if v > 1 then v = 1 end
				local i = (y + 1) * PADDED + x + 2
				img[i] = v
				if owner then
					local rgb: Vector3 = paletteNet[strokeColour[owner[y * SIZE + x + 1]] or 1]
					img[PS + i], img[2 * PS + i], img[3 * PS + i] = v * rgb.X, v * rgb.Y, v * rgb.Z
				end
			end
		end
		if checkpoint and y % 4 == 3 then checkpoint() end
	end
	return img
end

function M.rasterize(strokes, colours: { number }?): { number }?
	return rasterize(strokes, colours, nil)
end


M.rasterizeWithCheckpoints = rasterize
function M.beginRasterize(strokes,colours)
	return newJob(function(checkpoint) return rasterize(strokes,colours,checkpoint) end)
end
return M
end
