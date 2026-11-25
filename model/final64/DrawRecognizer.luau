--!native
--!optimize 2
-- DrawRecognizer: turns the player's strokes into the model-sized input and runs
-- the small CNN trained on Quick, Draw! (training/train.py). Runs on the server;
-- strokes arrive privately from their owner and are never sent to peers.
--
--   local R = require(ServerScriptService.DrawRecognizer)
--   local image = R.rasterize(strokes, colours) -- optional 1-based DrawPalette indices
--   local probs = R.forward(image)            -- array of probabilities, R.labels order
--   local top = R.recognize(strokes, 3, colours) -- old two-argument call stays valid
-- Colour models return four padded planes. The first plane is always ink, so the
-- existing 28px preview can use exactly its old indices in the fourth return value.
--
-- The rasterizer must stay identical to training/raster.py.
--
-- Speed: native code (--!native) only runs on servers and Android clients; PC, Mac and
-- iOS clients interpret this script. So the network math packs three output channels
-- into one Vector3: `acc += w * x` is one VM instruction doing three multiply-adds
-- (float32, like the network's weights), interpreted or native. Each conv row keeps a
-- 3-value sliding window in locals, and input channels that are all zero are skipped.

local Weights = require(script.Weights)

local M = {}
M.labels = Weights.labels

local SIZE: number = Weights.size or 28 -- older exports did not store their input resolution
local PADDED: number = SIZE + 2
M.size = SIZE
M.paddedSize = PADDED
M.modelName = Weights.name or "legacy"
M.qualityNormScale = Weights.qualityNormScale
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

-- base64 -> buffer -----------------------------------------------------------
local function decode64(s: string): buffer
	local map = {}
	local alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	for i = 1, 64 do
		map[string.byte(alphabet, i)] = i - 1
	end
	local n = #s
	local pad = 0
	if string.sub(s, -1) == "=" then pad += 1 end
	if string.sub(s, -2, -2) == "=" then pad += 1 end
	local out = buffer.create(n // 4 * 3 - pad)
	local o = 0
	for i = 1, n, 4 do
		local a, b, c, d = string.byte(s, i, i + 3)
		local v = bit32.bor(
			bit32.lshift(map[a], 18),
			bit32.lshift(map[b], 12),
			bit32.lshift(map[c] or 0, 6),
			map[d] or 0
		)
		buffer.writeu8(out, o, bit32.rshift(v, 16))
		if c ~= 61 then buffer.writeu8(out, o + 1, bit32.band(bit32.rshift(v, 8), 255)) end
		if d ~= 61 then buffer.writeu8(out, o + 2, bit32.band(v, 255)) end
		o += 3
	end
	return out
end

-- weights: int8 per output channel -> Vector3 groups of 3 output channels ------
-- layer.wv[g][k] = Vector3(w[3g-3][k], w[3g-2][k], w[3g-1][k]) (missing channels are 0)
-- layer.bv[g] = the matching biases. k runs over cin*9 (conv, ic-major then ky, kx) or nin (dense).
local layers = {}
do
	local buf = decode64(Weights.data)
	local off = 0
	local nlayers = #Weights.layers
	for li, spec in Weights.layers do
		local nout, nper = spec.nout, spec.nper
		local kind = spec.kind or (if li <= nlayers - 2 then "conv" else "dense")
		local w = table.create(nout * nper, 0)
		local qoff = off
		off += nout * nper
		local scaleInputs = if spec.scaleInputs then spec.cin else 1
		for o = 0, nout - 1 do
			for i = 1, nper do
				local ic = if scaleInputs > 1 then (i - 1) // 9 else 0
				local s = buffer.readf32(buf, off + (o * scaleInputs + ic) * 4)
				w[o * nper + i] = buffer.readi8(buf, qoff + o * nper + i - 1) * s
			end
		end
		off += nout * scaleInputs * 4
		local b = table.create(nout, 0)
		for o = 1, nout do
			b[o] = buffer.readf32(buf, off)
			off += 4
		end
		local G = (nout + 2) // 3
		local wv, bv = table.create(G), table.create(G)
		for g = 1, G do
			local o0, o1, o2 = (g - 1) * 3, (g - 1) * 3 + 1, (g - 1) * 3 + 2
			local arr = table.create(nper)
			for k = 1, nper do
				arr[k] = V3(
					w[o0 * nper + k],
					if o1 < nout then w[o1 * nper + k] else 0,
					if o2 < nout then w[o2 * nper + k] else 0
				)
			end
			wv[g] = arr
			bv[g] = V3(b[o0 + 1], if o1 < nout then b[o1 + 1] else 0, if o2 < nout then b[o2 + 1] else 0)
		end
		layers[li] = {
			kind = kind,
			normalizeInput = spec.normalizeInput or false,
			nout = nout,
			nper = nper,
			cin = if kind == "conv" then (spec.cin or nper // 9) else nper,
			pool = if kind == "conv" then (if spec.pool == nil then true else spec.pool) else false,
			groups = G,
			wv = wv,
			bv = bv,
		}
	end
	assert(off == buffer.len(buf), "weights size mismatch")
end

-- Reject a raster/weight mismatch at load, rather than silently reading only part
-- of a 40px feature map with a 28px dense layer.
do
	local h, cin = SIZE, INPUTS
	for _, layer in layers do
		if layer.kind == "conv" then
			assert(layer.cin == cin, "convolution channel mismatch")
			if layer.pool then h //= 2 end
		else
			assert(layer.nper == cin * h * h, "input resolution/dense shape mismatch")
			h = 1
		end
		cin = layer.nout
	end
	assert(cin == #M.labels, "output/label count mismatch")
end

-- Shared rasterizer is also used for the client preview.
local Rasterizer = require(script.DrawRasterizer)(Weights)
local rasterize = Rasterizer.rasterizeWithCheckpoints
function M.rasterize(strokes,colours) return rasterize(strokes,colours,nil) end

-- forward pass ---------------------------------------------------------------
local accPlane = table.create(SIZE * SIZE, Vector3.zero) -- reused accumulator plane (one group)

-- inp: cin planes of (H+2)x(W+2), zero border. live[ic] == false marks an all-zero plane.
-- Conv 3x3 (pad 1) + bias, then (optionally) 2x2 max-pool, then ReLU.
-- Returns 3*groups planes of (OH+2*opad)x(OW+2*opad) (zero border if opad == 1),
-- the matching live flags, OH and OW.
-- The type annotations matter: with them native code uses SIMD vector math (3x faster
-- than without); the interpreter ignores them.
local function convLayer(inp: { number }, live: { boolean }, H: number, W: number, layer, opad: number, workspaceAcc, checkpoint)
	local cin, G, wv, bv, pool = layer.cin, layer.groups, layer.wv, layer.bv, layer.pool
	local acc: { Vector3 } = workspaceAcc -- private for each suspended job
	local PW = W + 2
	local PS = (H + 2) * PW
	local HW = H * W
	local OH, OW = if pool then H // 2 else H, if pool then W // 2 else W
	local OPW = OW + 2 * opad
	local OPS = (OH + 2 * opad) * OPW
	local out = table.create(3 * G * OPS, 0)
	local outLive = table.create(3 * G, false)
	for g = 1, G do
		local wg: { Vector3 } = wv[g]
		local bias: Vector3 = bv[g]
		for i = 1, HW do
			acc[i] = bias
			if checkpoint and i % 256 == 0 then checkpoint() end
		end
		for ic = 0, cin - 1 do
			if not live[ic + 1] then continue end
			local k = ic * 9
			local w00: Vector3, w01: Vector3, w02: Vector3 = wg[k + 1], wg[k + 2], wg[k + 3]
			local w10: Vector3, w11: Vector3, w12: Vector3 = wg[k + 4], wg[k + 5], wg[k + 6]
			local w20: Vector3, w21: Vector3, w22: Vector3 = wg[k + 7], wg[k + 8], wg[k + 9]
			local ibase = ic * PS
			for y = 0, H - 1 do
				local r0 = ibase + y * PW + 1
				local r1 = r0 + PW
				local r2 = r1 + PW
				local a0: number, a1: number = inp[r0], inp[r0 + 1]
				local b0: number, b1: number = inp[r1], inp[r1 + 1]
				local c0: number, c1: number = inp[r2], inp[r2 + 1]
				local s0, s1, s2 = r0 + 1, r1 + 1, r2 + 1
				local base = y * W
				for x = 1, W do
					local a2: number, b2: number, c2: number = inp[s0 + x], inp[s1 + x], inp[s2 + x]
					local j = base + x
					acc[j] += w00 * a0 + w01 * a1 + w02 * a2
						+ w10 * b0 + w11 * b1 + w12 * b2
						+ w20 * c0 + w21 * c1 + w22 * c2
					a0, a1, b0, b1, c0, c1 = a1, a2, b1, b2, c1, c2
				end
				if checkpoint and y % 4 == 3 then checkpoint() end
			end
		end
		-- pool + ReLU + unpack the three channels into their planes
		local p0 = (g - 1) * 3 * OPS
		local p1, p2 = p0 + OPS, p0 + 2 * OPS
		local l0, l1, l2 = false, false, false
		for oy = 0, OH - 1 do
			local orow = (oy + opad) * OPW + opad + 1
			for ox = 0, OW - 1 do
				local vx, vy, vz
				if pool then
					local i = 2 * oy * W + 2 * ox + 1
					local p, q, r, s = acc[i], acc[i + 1], acc[i + W], acc[i + W + 1]
					vx, vy, vz = p.X, p.Y, p.Z
					local t = q.X
					if t > vx then vx = t end
					t = r.X
					if t > vx then vx = t end
					t = s.X
					if t > vx then vx = t end
					t = q.Y
					if t > vy then vy = t end
					t = r.Y
					if t > vy then vy = t end
					t = s.Y
					if t > vy then vy = t end
					t = q.Z
					if t > vz then vz = t end
					t = r.Z
					if t > vz then vz = t end
					t = s.Z
					if t > vz then vz = t end
				else
					local p = acc[oy * W + ox + 1]
					vx, vy, vz = p.X, p.Y, p.Z
				end
				local o = orow + ox
				if vx > 0 then
					out[p0 + o] = vx
					l0 = true
				end
				if vy > 0 then
					out[p1 + o] = vy
					l1 = true
				end
				if vz > 0 then
					out[p2 + o] = vz
					l2 = true
				end
			end
			if checkpoint and oy % 4 == 3 then checkpoint() end
		end
		outLive[(g - 1) * 3 + 1], outLive[(g - 1) * 3 + 2], outLive[(g - 1) * 3 + 3] = l0, l1, l2
	end
	return out, outLive, OH, OW
end

local function denseLayer(x: { number }, layer, relu: boolean, checkpoint)
	local nin, G, wv, bv = layer.nper, layer.groups, layer.wv, layer.bv
	local out = table.create(3 * G, 0)
	for g = 1, G do
		local wg: { Vector3 } = wv[g]
		local s: Vector3 = bv[g]
		for i = 1, nin do
			s += wg[i] * x[i]
			if checkpoint and i % 256 == 0 then checkpoint() end
		end
		local a, b, c = s.X, s.Y, s.Z
		if relu then
			if a < 0 then a = 0 end
			if b < 0 then b = 0 end
			if c < 0 then c = 0 end
		end
		local o = (g - 1) * 3
		out[o + 1], out[o + 2], out[o + 3] = a, b, c
	end
	return out
end

-- image: padded model-sized array from rasterize. Returns logits (length #labels).
local function logits(image: { number }, workspaceAcc, checkpoint): { number }
	local x = image
	if INPUTS == 4 and #image == PS then
		x = table.create(INPUTS * PS, 0)
		for i = 1, PS do
			x[i] = image[i]
			if checkpoint and i % 256 == 0 then checkpoint() end
		end
	end
	assert(#x >= INPUTS * PS, "input plane count mismatch")
	local live = { true }
	for ic = 1, INPUTS - 1 do
		local nonzero = false
		for i = ic * PS + 1, (ic + 1) * PS do
			if x[i] ~= 0 then nonzero = true break end
			if checkpoint and i % 256 == 0 then checkpoint() end
		end
		live[ic + 1] = nonzero
	end
	local H, W = SIZE, SIZE
	local n = #layers
	local qualityNorm = nil
	for li, layer in layers do
		if layer.kind == "conv" then
			local nextIsConv = li < n and layers[li + 1].kind == "conv"
			x, live, H, W = convLayer(x, live, H, W, layer, if nextIsConv then 1 else 0, workspaceAcc, checkpoint)
		else
			if layer.normalizeInput then
                local sum = 0
                for i = 1, layer.nper do sum += x[i] * x[i] end
                qualityNorm = math.sqrt(sum)
                local inv = 1 / math.max(qualityNorm, 1e-8)
                for i = 1, layer.nper do
                    x[i] *= inv
                    if checkpoint and i % 256 == 0 then checkpoint() end
                end
            end
            x = denseLayer(x, layer, li < n, checkpoint)
		end
	end
	local nout = layers[n].nout
	local z = table.create(nout, 0)
	for i = 1, nout do
		z[i] = x[i]
	end
	return z, qualityNorm
end

function M.logits(image: { number }): { number }
	return logits(image, accPlane, nil)
end

local function forward(image: { number }, workspaceAcc, checkpoint): { number }
	local z, qualityNorm = logits(image, workspaceAcc, checkpoint)
	local m = -math.huge
	for _, v in z do
		if v > m then m = v end
	end
	local sum = 0
	local p = table.create(#z, 0)
	for i, v in z do
		local e = math.exp(v - m)
		p[i] = e
		sum += e
	end
	for i = 1, #p do
		p[i] /= sum
	end
	return p, qualityNorm
end

function M.forward(image: { number }): { number }
	return forward(image, accPlane, nil)
end

-- Call step once per Heartbeat. The budget is active CPU time, not a sleep or an
-- artificial delay. Timing includes resume/checkpoint overhead and allocations.
-- A row/checkpoint can slightly overrun the requested budget; maxSliceMs measures
-- that overrun. Every job owns its accumulator, so interleaved jobs/sync calls
-- cannot corrupt a suspended forward pass. The old synchronous API is unchanged.
local newJob = require(script.DrawRasterizer.DrawJob)

function M.beginForward(image: { number })
	-- Caller owns image until completion; recognize jobs create their own image.
	return newJob(function(checkpoint)
		local workspaceAcc = table.create(SIZE * SIZE, Vector3.zero)
		return forward(image, workspaceAcc, checkpoint)
	end)
end

function M.beginRecognize(strokes, k: number?, colours: { number }?)
	-- Finished stroke point lists must be immutable until completion/cancellation.
	return newJob(function(checkpoint, activeClock)
		local t0 = activeClock()
		local image = rasterize(strokes, colours, checkpoint)
		local t1 = activeClock()
		if not image then return { top = {}, rasterMs = t1 - t0, networkMs = 0, image = nil } end
		local p, qualityNorm = forward(image, table.create(SIZE * SIZE, Vector3.zero), checkpoint)
		local t2 = activeClock()
		checkpoint()
		local idx = table.create(#p, 0)
		for i = 1, #p do idx[i] = i end
		table.sort(idx, function(a, b) return p[a] > p[b] end)
		local top = {}
		for i = 1, math.min(k or 3, #idx) do
			top[i] = { label = M.labels[idx[i]], prob = p[idx[i]], index = idx[i] }
		end
		return { top = top, rasterMs = t1 - t0, networkMs = t2 - t1, image = image, qualityNorm = qualityNorm }
	end)
end

-- Convenience: strokes -> top-k { {label, prob, index} }, rasterMs, netMs, image
function M.recognize(strokes, k: number?, colours: { number }?)
	local t0 = os.clock()
	local image = M.rasterize(strokes, colours)
	if not image then
		return {}, 0, 0
	end
	local t1 = os.clock()
	local p, qualityNorm = M.forward(image)
	local t2 = os.clock()
	local idx = table.create(#p, 0)
	for i = 1, #p do
		idx[i] = i
	end
	table.sort(idx, function(a, b)
		return p[a] > p[b]
	end)
	local top = {}
	for i = 1, math.min(k or 3, #idx) do
		top[i] = { label = M.labels[idx[i]], prob = p[idx[i]], index = idx[i] }
	end
	return top, (t1 - t0) * 1000, (t2 - t1) * 1000, image, qualityNorm
end

return M
