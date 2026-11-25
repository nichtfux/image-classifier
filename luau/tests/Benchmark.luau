-- Dev-only. Pass separately loaded native and interpreted recognizers, plus the
-- reference vectors for the same model. No reference drawing is displayed.
return function(native, interpreted, vectors)
	assert(native.size == interpreted.size and #native.labels == #interpreted.labels)
	local function strokes(sample)
		local result = {}
		for _, flat in sample.strokes do
			local st = {}
			for i = 1, #flat, 2 do st[#st + 1] = Vector2.new(flat[i], flat[i + 1]) end
			result[#result + 1] = st
		end
		return result
	end
	local function timed(recognizer, image)
		local t0 = os.clock()
		local p = recognizer.forward(image)
		return (os.clock() - t0) * 1000, p
	end
	for i = 1, 6 do
		local s = vectors[math.floor((i - 1) * (#vectors - 1) / 5) + 1]
		local image = native.rasterize(strokes(s), s.colours)
		native.forward(image)
		interpreted.forward(image)
	end
	local nt, it = {}, {}
	local maxProbabilityDiff, agree = 0, 0
	for n, sample in vectors do
		local image = native.rasterize(strokes(sample), sample.colours)
		local pn, pi
		if n % 2 == 0 then
			it[n], pi = timed(interpreted, image)
			nt[n], pn = timed(native, image)
		else
			nt[n], pn = timed(native, image)
			it[n], pi = timed(interpreted, image)
		end
		local bn, bi = 1, 1
		for j = 1, #pn do
			maxProbabilityDiff = math.max(maxProbabilityDiff, math.abs(pn[j] - pi[j]))
			if pn[j] > pn[bn] then bn = j end
			if pi[j] > pi[bi] then bi = j end
		end
		if bn == bi then agree += 1 end
		if n % 5 == 0 then task.wait() end
	end
	assert(agree == #vectors, "native/interpreted prediction mismatch")
	assert(maxProbabilityDiff < 1e-3, "native/interpreted probability mismatch")
	local function stats(times)
		table.sort(times)
		return { median = times[math.ceil(#times / 2)], min = times[1],
			p95 = times[math.ceil(#times * 0.95)], max = times[#times] }
	end
	return { model = native.modelName, size = native.size, categories = #native.labels,
		samples = #vectors, native_ms = stats(nt), interpreted_ms = stats(it),
		prediction_agreement = agree, max_probability_diff = maxProbabilityDiff }
end
