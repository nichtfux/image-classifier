--!native
-- Server-only admission, bounded round-robin work and one-use trusted results.
-- No drawing or result is written to an Instance/attribute or sent to peers.
return function(R, Protocol, newJob, options)
	options = options or {}
	local clock = options.clock or os.clock
	local reply = options.reply or function() end
	local authorize = options.authorize or function() return true end -- NN test has no rounds yet
	local FRAME_MS, QUEUE_MAX, DEADLINE = 2,32,6
	local queue,states = {},{}
	local service = { metrics = { accepted=0,completed=0,rejected=0,failed=0,maxFrameMs=0,maxSliceMs=0,maxDepth=0 } }
	local function send(player,id,packet,mode)
		packet.mode=mode or "submit"
		local ok,err=pcall(reply,player,id,packet)
		if not ok then warn("[Recognition] reply failed: "..tostring(err)) end
	end
	local function reject(player,id,code,retry,mode)
		service.metrics.rejected += 1
		send(player,id,{ok=false,error=code,retryAfter=retry or 1},mode)
		return false,code
	end
	local function finish(entry,result,err)
		local s = states[entry.player]
		if not s or s.entry ~= entry then return end
		s.entry = nil
		if err then
			service.metrics.failed += 1
			send(entry.player,entry.id,{ok=false,error=err,retryAfter=1},entry.mode)
			return
		end
		service.metrics.completed += 1
		local stats = entry.job.stats
		local decode = entry.decodeStats
		local wall = (clock()-entry.acceptedAt)*1000
		local active = stats.activeMs+decode.activeMs
		local report = {
			server=true,activeMs=active,wallMs=wall,
			queueWaitMs=math.max(0,wall-active),firstSliceWaitMs=(entry.startedAt-entry.acceptedAt)*1000,decodeMs=decode.activeMs,
			maxSliceMs=math.max(stats.maxSliceMs,decode.maxSliceMs),slices=stats.slices+decode.slices,
			payloadBytes=entry.payloadBytes,networkMs=result.networkMs,rasterMs=result.rasterMs,
		}
		-- Separate server table; a client receives only its serialized copy.
		if entry.mode=="submit" then
			s.trusted = {id=entry.id,createdAt=clock(),top=result.top,model=R.modelName}
			if options.completed then task.defer(options.completed,entry.player,entry.id) end
		end
		send(entry.player,entry.id,{ok=true,top=result.top,model=R.modelName,stats=report},entry.mode)
	end
	function service:request(player,id,payload,trustResult)
		local now=clock()
		local mode=if trustResult==false then "preview" else "submit"
		local s=states[player]
		if not s then s={nextAttempt=0,nextGuess=0,lastIds={}};states[player]=s end
		-- Even malformed attempts consume this cheap ingress gate. Silently drop
		-- excess spam, avoiding response amplification or per-attempt allocations.
		if now < s.nextAttempt then return false,"ingress_rate" end
		s.nextAttempt=now+.25
		if type(id)~="number" or id~=id or id%1~=0 or id<1 or id>2147483647 then return false,"request_id" end
		if id<=(s.lastIds[mode] or 0) then return reject(player,id,"stale_request",nil,mode) end
		s.lastIds[mode]=id
		if not authorize(player) then return reject(player,id,"not_allowed",nil,mode) end
		if s.entry then return reject(player,id,"in_flight",nil,mode) end
		if now < s.nextGuess then return reject(player,id,"rate_limit",s.nextGuess-now,mode) end
		if typeof(payload)~="buffer" then return reject(player,id,"payload_type",nil,mode) end
		local length=buffer.len(payload)
		if length<12 or length>Protocol.MAX_BYTES then return reject(player,id,"payload_size",nil,mode) end
		if #queue>=QUEUE_MAX then return reject(player,id,"server_busy",nil,mode) end
		s.nextGuess=now+1
		if mode=="submit" then s.trusted=nil end
		local entry={player=player,id=id,mode=mode,acceptedAt=now,deadline=now+DEADLINE,payloadBytes=length,phase="decode"}
		entry.job=newJob(function(checkpoint)
			local decoded,err=Protocol.decode(payload,checkpoint)
			return {decoded=decoded,error=err}
		end)
		s.entry=entry
		queue[#queue+1]=entry
		self.metrics.accepted+=1
		self.metrics.maxDepth=math.max(self.metrics.maxDepth,#queue)
		return true
	end
	function service:tick()
		local start=os.clock()
		local endAt=start+FRAME_MS/1000
		local resumes=0
		while #queue>0 and os.clock()<endAt and resumes<16 do
			local entry=table.remove(queue,1)
			local s=states[entry.player]
			if not s or s.entry~=entry then entry.job:cancel();continue end
			if clock()>=entry.deadline then entry.job:cancel();finish(entry,nil,"timeout");continue end
			entry.startedAt=entry.startedAt or clock()
			local budget=math.min(.75,math.max(.01,(endAt-os.clock())*1000))
			local ok,done,result,stats=pcall(entry.job.step,entry.job,budget)
			resumes+=1
			if not ok then
				warn("[Recognition] inference failed: "..tostring(done))
				finish(entry,nil,"recognition_failed")
			elseif done and entry.phase=="decode" then
				if result.error then finish(entry,nil,result.error) else
					entry.decodeStats=stats
					entry.phase="recognize"
					entry.job=R.beginRecognize(result.decoded.strokes,3,result.decoded.colours)
					queue[#queue+1]=entry
				end
			elseif done then finish(entry,result,nil)
			else queue[#queue+1]=entry end
			if stats then self.metrics.maxSliceMs=math.max(self.metrics.maxSliceMs,stats.maxSliceMs) end
		end
		self.metrics.maxFrameMs=math.max(self.metrics.maxFrameMs,(os.clock()-start)*1000)
	end
	function service:consume(player,id)
		local s=states[player];local result=s and s.trusted
		if not result or result.id~=id or clock()-result.createdAt>10 or not authorize(player) then return nil end
		s.trusted=nil -- one-use; gameplay must still enforce its round/cost/cooldown
		return result
	end
	function service:remove(player)
		local s=states[player]
		if s and s.entry then s.entry.job:cancel() end
		states[player]=nil
		for i=#queue,1,-1 do if queue[i].player==player then table.remove(queue,i) end end
	end
	function service:depth() return #queue end
	function service:isInFlight(player) return states[player] and states[player].entry~=nil or false end
	function service:invalidateTrusted(player)
		if states[player] then states[player].trusted=nil end
	end
	return service
end
