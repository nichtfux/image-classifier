-- Dev-only test harness. No drawing is rendered, replicated or sent.
-- Call with fresh native/interpreted recognizers. waitFrame=true measures real
-- Heartbeat wall latency; false verifies all vectors without waiting hundreds of
-- real frames. Run the frame test separately on a varied small subset.
return function(R, V, waitFrame: boolean?, budgetMs: number?)
	local budget = budgetMs or 1.25
	local report = { model = R.modelName, budgetMs = budget, samples = #V,
		maxProbabilityDiff = 0, maxSliceMs = 0, maxExternalStepMs = 0,
		maxWallMs = 0, maxActiveMs = 0, slices = 0 }
	local function run(job)
		while true do
			local t = os.clock()
			local done, value, stats = job:step(budget)
			report.maxExternalStepMs = math.max(report.maxExternalStepMs, (os.clock() - t) * 1000)
			if done then
				report.maxSliceMs = math.max(report.maxSliceMs, stats.maxSliceMs)
				report.maxWallMs = math.max(report.maxWallMs, stats.wallMs)
				report.maxActiveMs = math.max(report.maxActiveMs, stats.activeMs)
				report.slices += stats.slices
				return value
			end
			if waitFrame then game:GetService("RunService").Heartbeat:Wait() end
		end
	end
	local first, second
	for n,s in V do
		local strokes = {}
		for _,flat in s.strokes do
			local st = {}
			for i=1,#flat,2 do st[#st+1] = Vector2.new(flat[i],flat[i+1]) end
			strokes[#strokes+1] = st
		end
		local image = R.rasterize(strokes,s.colours)
		first, second = first or image, image
		local expected = R.forward(image)
		local got = run(R.beginForward(image))
		for i=1,#expected do report.maxProbabilityDiff = math.max(report.maxProbabilityDiff,math.abs(expected[i]-got[i])) end
		local rec = run(R.beginRecognize(strokes,3,s.colours))
		local syncTop = R.recognize(strokes,3,s.colours)
		for i=1,3 do
			assert(rec.top[i].index==syncTop[i].index, "sliced recognize rank mismatch")
			assert(rec.top[i].prob==syncTop[i].prob, "sliced recognize probability mismatch")
		end
		for i=1,#image do assert(image[i]==rec.image[i], "sliced raster mismatch") end
		assert(rec.networkMs>=0 and rec.rasterMs>=0, "negative CPU accounting")
		if n%5==0 then task.wait() end
	end
	assert(report.maxProbabilityDiff==0, "yield changed network arithmetic")
	-- Suspended jobs must also survive other jobs and synchronous calls.
	if first and second then
		local a,b = R.beginForward(first),R.beginForward(second)
		a:step(.001);b:step(.001)
		R.forward(first)
		local rounds = 0
		while a.state~="completed" or b.state~="completed" do
			if a.state~="completed" then a:step(.5) end
			if b.state~="completed" then b:step(.5) end
			rounds += 1
			if rounds%8==0 then task.wait() end
		end
		for _,pair in {{a,first},{b,second}} do
			local expected = R.forward(pair[2])
			for i,p in expected do assert(p==pair[1].result[i], "shared scratch corrupted a suspended job") end
		end
		local cancelled = R.beginForward(first)
		cancelled:step(.001);cancelled:cancel()
		local done,value = cancelled:step()
		assert(done and value==nil and cancelled.state=="cancelled", "cancel published a result")
	end
	local empty = run(R.beginRecognize({},3))
	assert(#empty.top==0 and empty.image==nil, "empty sliced drawing")
	return report
end
