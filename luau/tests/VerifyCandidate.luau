-- Private Studio integration/parity check; never displays training drawings.
-- Existing project Verify modules use this explicit assertion/return convention.
return function(folder)
	local RunService=game:GetService("RunService")
	assert(RunService:IsStudio() and RunService:IsServer(),"run on the Studio server during Play")
	local R=require(folder.DrawRecognizer:Clone())
	local vectors=require(folder.TestVectors:Clone())
	assert(#vectors>=11,"missing candidate parity vectors")
	local worstImage,worstLogit,topMismatch,probError=0,0,0,0
	local firstImage
	for _,v in vectors do
		assert(v.size==R.size and #v.logits==#R.labels,"model/vector layout mismatch")
		local strokes={}
		for _,flat in v.strokes do
			local stroke={}
			for i=1,#flat,2 do stroke[#stroke+1]=Vector2.new(flat[i],flat[i+1]) end
			strokes[#strokes+1]=stroke
		end
		local image=R.rasterize(strokes,v.colours)
		firstImage=firstImage or image
		for c=1,R.inputChannels do
			local sum,moment=0,0
			for y=0,R.size-1 do for x=0,R.size-1 do
				local value=image[(c-1)*R.planeSize+(y+1)*R.paddedSize+x+2]
				sum+=value;moment+=value*(y*R.size+x)
			end end
			worstImage=math.max(worstImage,math.abs(sum-v.imageSums[c]),math.abs(moment-v.imageMoments[c])/R.size^2)
		end
		local z=R.logits(image)
		local actual,reference=1,1
		for i=1,#z do
			worstLogit=math.max(worstLogit,math.abs(z[i]-v.logits[i]))
			if z[i]>z[actual] then actual=i end
			if v.logits[i]>v.logits[reference] then reference=i end
		end
		if actual~=reference then topMismatch+=1 end
		task.wait()
	end
	assert(worstImage<1e-8,"Python/Studio raster mismatch")
	assert(worstLogit<3e-4 and topMismatch==0,"Python/Studio network mismatch")
	local synchronous=R.forward(firstImage)
	local job=R.beginForward(firstImage)
	while not job:step(.75) do RunService.Heartbeat:Wait() end
	for i=1,#synchronous do probError=math.max(probError,math.abs(synchronous[i]-job.result[i])) end
	assert(probError<1e-6,"time-sliced forward differs")
	local forward,recognize={},{ }
	local strokes={{Vector2.new(80,80),Vector2.new(320,80),Vector2.new(320,320),Vector2.new(80,320),Vector2.new(80,80)}}
	for i=1,21 do
		local t=os.clock();R.forward(firstImage);forward[i]=(os.clock()-t)*1000
		t=os.clock();R.recognize(strokes,3,{1});recognize[i]=(os.clock()-t)*1000
		RunService.Heartbeat:Wait()
	end
	table.sort(forward);table.sort(recognize)
	return {status="passed",model=R.modelName,vectors=#vectors,maxImageDiff=worstImage,
		maxLogitDiff=worstLogit,topMismatches=topMismatch,slicedProbabilityDiff=probError,
		forwardMedianMs=forward[11],forwardP95Ms=forward[20],recognizeMedianMs=recognize[11],
		maxSliceMs=job.stats.maxSliceMs,slicedActiveMs=job.stats.activeMs,slicedWallMs=job.stats.wallMs}
end
