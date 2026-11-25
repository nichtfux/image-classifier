-- Server-owned drawing sessions. The client cannot open a session, choose its
-- context, or obtain a second evaluation for an accepted finished drawing.
return function(queue,options)
	local clock=options.clock or os.clock
	local sessions,attempts,opened={},{},{}
	local gate={queue=queue}
	function gate:openDrawing(player,context,lifetime)
		local ttl=lifetime or 60
		assert(type(ttl)=="number" and ttl==ttl and ttl>=1 and ttl<=120,"drawing lifetime must be 1-120 seconds")
		if queue:isInFlight(player) then return nil,"in_flight" end
		local now=clock()
		if opened[player] and now-opened[player]<1 then return nil,"rate_limit" end
		opened[player]=now
		queue:invalidateTrusted(player)
		local token=options.issue()
		sessions[player]={token=token,expires=now+ttl,context=context}
		return token
	end
	local function reject(player,id,error)
		options.reply(player,id,{ok=false,error=error,mode="submit"})
		return false,error
	end
	function gate:submit(player,id,token,payload)
		local now=clock()
		if now<(attempts[player] or 0) then return false,"ingress_rate" end
		attempts[player]=now+.25
		if type(id)~="number" or id~=id or id%1~=0 or id<1 or id>2147483647 then return false,"request_id" end
		local s=sessions[player]
		if type(token)~="string" or #token>64 or not s or s.token~=token or now>s.expires then
			return reject(player,id,"drawing_not_open")
		end
		local ok,err=queue:request(player,id,payload,true)
		if ok then s.token=nil;s.requestId=id end -- consume permission at admission
		return ok,err
	end
	function gate:consume(player,id)
		local s=sessions[player]
		if not s or s.requestId~=id then return nil end
		local result=queue:consume(player,id)
		if not result then return nil end
		result.context=s.context -- original server-owned round/ability context
		sessions[player]=nil
		return result
	end
	function gate:remove(player)
		sessions[player]=nil;attempts[player]=nil;opened[player]=nil;queue:remove(player)
	end
	return gate
end
