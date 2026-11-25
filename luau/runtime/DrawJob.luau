-- Shared cooperative task; no model or weights.
local function newJob(work)
	local job = {
		state = "pending",
		result = nil,
		stats = { activeMs = 0, wallMs = 0, maxSliceMs = 0, slices = 0 },
	}
	local started, sliceStart, deadline = nil, 0, 0
	local function activeClock()
		return job.stats.activeMs + (os.clock() - sliceStart) * 1000
	end
	local function checkpoint()
		if os.clock() >= deadline then coroutine.yield() end
	end
	local thread = coroutine.create(function() return work(checkpoint, activeClock) end)
	function job:step(budgetMs: number?)
		if self.state == "completed" or self.state == "cancelled" then
			return true, self.result, self.stats
		end
		assert(self.state ~= "failed", "cannot resume failed inference")
		local budget = budgetMs or 1.25
		assert(budget > 0 and budget < math.huge, "invalid slice budget")
		sliceStart = os.clock()
		started = started or sliceStart
		deadline = sliceStart + budget / 1000
		self.state = "running"
		local ok, value = coroutine.resume(thread)
		local now = os.clock()
		local elapsed = (now - sliceStart) * 1000
		self.stats.activeMs += elapsed
		self.stats.maxSliceMs = math.max(self.stats.maxSliceMs, elapsed)
		self.stats.wallMs = (now - started) * 1000
		self.stats.slices += 1
		if not ok then
			self.state = "failed"
			thread = nil
			error(value, 0)
		end
		if coroutine.status(thread) == "dead" then
			self.state = "completed"
			self.result = value
			thread = nil
			return true, value, self.stats
		end
		return false, nil, self.stats
	end
	function job:cancel()
		if self.state == "completed" or self.state == "failed" then return end
		self.state = "cancelled"
		thread = nil -- release suspended arrays; never publish a stale result
	end
	return job
end


return newJob
