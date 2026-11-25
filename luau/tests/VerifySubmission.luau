-- Dev-only finished-drawing permission/replay checks, no UI or dataset rendering.
return function(P,newJob,createQueue,createGate)
	local now,calls,issued,completed=100,0,0,0
	local replies={}
	local function reply(player,id,result) replies[#replies+1]=result end
	local R={modelName="test",beginRecognize=function()
		calls+=1
		return newJob(function() return {top={{label="test",prob=1,index=1}},rasterMs=0,networkMs=0} end)
	end}
	local queue=createQueue(R,P,newJob,{clock=function() return now end,reply=reply,completed=function() completed+=1 end})
	local gate=createGate(queue,{clock=function() return now end,reply=reply,issue=function() issued+=1;return "ticket"..issued end})
	local bytes=P.encode({{Vector2.zero,Vector2.new(10,20)}},{1},{4})
	assert(not gate:submit("p",1,"made-up",bytes),"client created a drawing session")
	now+=.3
	local token=gate:openDrawing("p",{round=7},5)
	assert(token,"server did not open drawing")
	assert(gate:submit("p",1,token,bytes),"finished submission rejected")
	now+=.3
	assert(not gate:submit("p",2,token,bytes),"ticket allowed two submissions")
	for i=1,10 do queue:tick() end
	task.wait()
	assert(completed==1,"finished drawing completion missing")
	assert(calls==1,"drawing evaluated more than once")
	local result=gate:consume("p",1)
	assert(result and result.context.round==7,"trusted server context lost")
	assert(gate:consume("p",1)==nil,"result consumed twice")
	now+=2
	local expired=gate:openDrawing("p",{},1)
	now+=2
	assert(not gate:submit("p",3,expired,bytes),"expired permission accepted")
	assert(calls==1,"expired drawing reached model")
	now+=2
	assert(queue:request("p",1,bytes,false),"preview should have independent ids")
	for i=1,10 do queue:tick() end
	task.wait()
	assert(completed==1,"Studio preview emitted game completion")
	assert(queue:consume("p",1)==nil,"Studio preview became game authority")
	assert(calls==2,"preview did not use same inference queue")
	gate:remove("p")
	assert(not gate:submit("p",4,expired,bytes),"disconnect retained permission")
	return {status="passed",finishedEvaluations=1,previewTrusted=false,oneUseResult=true,serverContextPreserved=true}
end
