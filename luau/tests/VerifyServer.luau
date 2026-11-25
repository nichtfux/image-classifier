-- Dev-only checks; never displays or transmits a training drawing.
return function(P,newJob,createQueue,R,Preview)
	local checks=0
	local function check(value,message) assert(value,message);checks+=1 end
	local strokes={{Vector2.new(10.013,20.027),Vector2.new(300.032,400.01)}}
	local bytes,canonical=P.encode(strokes,{4},{16})
	check(bytes~=nil,"encode valid input")
	local decoded=P.decode(bytes)
	check(decoded.strokes[1][1]==canonical.strokes[1][1],"quantized point agreement")
	check(decoded.colours[1]==4 and decoded.widths[1]==16,"ink metadata")
	check(math.abs(decoded.strokes[1][1].X-strokes[1][1].X)<=1/32,"quantization tolerance")
	local a=R.rasterize(decoded.strokes,decoded.colours)
	local b=Preview.rasterize(canonical.strokes,canonical.colours)
	for i=1,#a do assert(a[i]==b[i],"preview/server raster difference") end
	checks+=1
	local function copy(payload)
		local out=buffer.create(buffer.len(payload));buffer.copy(out,0,payload);return out
	end
	local invalid={}
	local function mutate(offset,value,wide)
		local out=copy(bytes)
		if wide then buffer.writeu16(out,offset,value) else buffer.writeu8(out,offset,value) end
		invalid[#invalid+1]=out
	end
	mutate(0,2);mutate(1,1);mutate(2,0,true);mutate(2,129,true)
	mutate(4,0);mutate(4,11);mutate(5,2);mutate(6,0,true);mutate(6,2049,true);mutate(8,7041,true)
	local truncated=buffer.create(12);buffer.copy(truncated,0,bytes,0,12);invalid[#invalid+1]=truncated
	local trailing=buffer.create(buffer.len(bytes)+1);buffer.copy(trailing,0,bytes);invalid[#invalid+1]=trailing
	local overflow=buffer.create(4+3*4+4097*4)
	buffer.writeu8(overflow,0,1);buffer.writeu16(overflow,2,3)
	local offset=4
	for _,n in {2048,2048,1} do
		buffer.writeu8(overflow,offset,1);buffer.writeu8(overflow,offset+1,4);buffer.writeu16(overflow,offset+2,n)
		offset+=4+n*4
	end
	invalid[#invalid+1]=overflow
	local zig,cols,widths={},{},{}
	for i=1,128 do local pts={};for j=1,32 do pts[j]=if j%2==0 then Vector2.new(440,440) else Vector2.zero end;zig[i]=pts;cols[i]=1;widths[i]=4 end
	invalid[#invalid+1]=P.encode(zig,cols,widths) -- bounded byte count, excessive repeated ink
	for _,payload in invalid do check(P.decode(payload)==nil,"malformed buffer accepted") end
	check(P.decode({})==nil,"table payload accepted")
	check(P.decode(buffer.create(P.MAX_BYTES+1))==nil,"oversized payload accepted")
	check(P.encode({{Vector2.new(0/0,2)}})==nil,"NaN accepted")
	check(P.encode({{Vector2.new(math.huge,2)}})==nil,"infinity accepted")
	check(P.encode({{Vector2.new(-1,2)}})==nil,"negative coordinate accepted")
	local many={}
	for i=1,129 do many[i]={Vector2.zero} end
	check(P.encode(many)==nil,"stroke cap bypass")
	local points=table.create(2048,Vector2.zero)
	local maxBytes=P.encode({points,points},{1,10},{4,16})
	check(maxBytes~=nil and P.decode(maxBytes).points==4096,"maximum legal point budget")
	local now,replies,modelCalls=100,{},0
	local fake={modelName="test"}
	function fake.beginRecognize()
		modelCalls+=1
		return newJob(function() return {top={{label="test",prob=1,index=1}},rasterMs=0,networkMs=0} end)
	end
	local function create()
		return createQueue(fake,P,newJob,{clock=function() return now end,reply=function(player,id,result) replies[#replies+1]={player=player,id=id,result=result} end})
	end
	local service=create()
	for i,payload in invalid do check(service:request("bad"..i,1,payload),"bad input should reach bounded decode") end
	for i=1,20 do service:tick() end
	check(service:depth()==0 and modelCalls==0,"invalid input reached model")
	check(service:request("one",1,bytes),"valid admission")
	check(not service:request("one",2,bytes),"ingress limit bypass")
	now+=.3
	local ok,err=service:request("one",2,bytes)
	check(not ok and err=="in_flight","one in-flight rule")
	for i=1,10 do service:tick() end
	check(modelCalls==1,"valid drawing not inferred exactly once")
	local result=service:consume("one",1)
	check(result and result.top[1].label=="test","trusted server result missing")
	check(service:consume("one",1)==nil,"trusted result reused")
	now+=.3
	ok,err=service:request("one",3,bytes)
	check(not ok and err=="rate_limit","accepted rate limit bypass")
	now+=1
	check(service:request("one",4,bytes),"cooldown did not recover")
	for i=1,10 do service:tick() end
	now+=11
	check(service:consume("one",4)==nil,"expired trusted result")
	service=create()
	for i=1,32 do check(service:request("burst"..i,1,bytes),"legal queue admission") end
	check(not service:request("overflow",1,bytes),"global queue cap bypass")
	service:remove("burst1")
	check(service:depth()==31,"disconnect retained work")
	now+=7
	service:tick()
	check(service:depth()==0,"deadline did not clear queue")
	check(service:consume("burst2",1)==nil,"timed-out work trusted")
	return {checks=checks,malformedBuffers=#invalid,previewPixelDiff=0,queueDepth=service:depth(),status="passed"}
end
