--!strict
-- One bounded binary request, no nested client tables. All limits are enforced
-- again by decode on the server. Coordinates are canonical 440px canvas units.
local P = {
	VERSION = 1, CANVAS = 440, QUANT = 16,
	MAX_STROKES = 128, MAX_POINTS = 4096, MAX_STROKE_POINTS = 2048,
	MAX_INK_SPANS = 200, -- total Manhattan path length / drawing bbox span
	MAX_BYTES = 4 + 128 * 4 + 4096 * 4,
}
local function integer(n, lo, hi)
	return type(n) == "number" and n == n and n % 1 == 0 and n >= lo and n <= hi
end

function P.encode(strokes, colours, widths)
	if type(strokes) ~= "table" or #strokes < 1 or #strokes > P.MAX_STROKES then return nil, "stroke_limit" end
	local total = 0
	for i, stroke in strokes do
		if type(stroke) ~= "table" or #stroke < 1 or #stroke > P.MAX_STROKE_POINTS then return nil, "point_limit" end
		total += #stroke
		if total > P.MAX_POINTS then return nil, "point_limit" end
		local colour, width = (colours and colours[i]) or 1, (widths and widths[i]) or 4
		if not integer(colour,1,10) or (width ~= 4 and width ~= 9 and width ~= 16) then return nil, "ink_metadata" end
		for _, p in stroke do
			if typeof(p) ~= "Vector2" or not (p.X >= 0 and p.X <= P.CANVAS and p.Y >= 0 and p.Y <= P.CANVAS) then return nil, "coordinates" end
		end
	end
	local bytes = buffer.create(4 + #strokes * 4 + total * 4)
	buffer.writeu8(bytes,0,P.VERSION); buffer.writeu8(bytes,1,0); buffer.writeu16(bytes,2,#strokes)
	local offset, canonical = 4, { strokes = {}, colours = {}, widths = {}, points = total }
	for i, stroke in strokes do
		local colour, width = (colours and colours[i]) or 1, (widths and widths[i]) or 4
		buffer.writeu8(bytes,offset,colour); buffer.writeu8(bytes,offset+1,width); buffer.writeu16(bytes,offset+2,#stroke)
		offset += 4
		local points = table.create(#stroke)
		for j, p in stroke do
			local x,y = math.floor(p.X * P.QUANT + .5), math.floor(p.Y * P.QUANT + .5)
			buffer.writeu16(bytes,offset,x); buffer.writeu16(bytes,offset+2,y); offset += 4
			points[j] = Vector2.new(x/P.QUANT,y/P.QUANT)
		end
		canonical.strokes[i],canonical.colours[i],canonical.widths[i] = points,colour,width
	end
	return bytes,canonical
end

function P.decode(bytes, checkpoint)
	if typeof(bytes) ~= "buffer" then return nil, "payload_type" end
	local length = buffer.len(bytes)
	if length < 12 or length > P.MAX_BYTES then return nil, "payload_size" end
	if buffer.readu8(bytes,0) ~= P.VERSION or buffer.readu8(bytes,1) ~= 0 then return nil, "version" end
	local count = buffer.readu16(bytes,2)
	if count < 1 or count > P.MAX_STROKES then return nil, "stroke_limit" end
	local offset,total,out = 4,0,{strokes={},colours={},widths={}}
	local minx,miny,maxx,maxy,path = math.huge,math.huge,-math.huge,-math.huge,0
	for i = 1,count do
		if offset+4 > length then return nil, "truncated" end
		local colour,width,n = buffer.readu8(bytes,offset),buffer.readu8(bytes,offset+1),buffer.readu16(bytes,offset+2)
		offset += 4
		if colour < 1 or colour > 10 or (width ~= 4 and width ~= 9 and width ~= 16) then return nil, "ink_metadata" end
		total += n
		if n < 1 or n > P.MAX_STROKE_POINTS or total > P.MAX_POINTS then return nil, "point_limit" end
		if offset+n*4 > length then return nil, "truncated" end
		local points = table.create(n)
		local lastx,lasty
		for j = 1,n do
			local x,y = buffer.readu16(bytes,offset),buffer.readu16(bytes,offset+2); offset += 4
			if x > P.CANVAS*P.QUANT or y > P.CANVAS*P.QUANT then return nil, "coordinates" end
			minx=math.min(minx,x);miny=math.min(miny,y);maxx=math.max(maxx,x);maxy=math.max(maxy,y)
			if lastx then path+=math.abs(x-lastx)+math.abs(y-lasty) end
			lastx,lasty=x,y
			points[j] = Vector2.new(x/P.QUANT,y/P.QUANT)
			if checkpoint and j%128==0 then checkpoint() end
		end
		out.strokes[i],out.colours[i],out.widths[i] = points,colour,width
		if checkpoint then checkpoint() end
	end
	if offset ~= length then return nil, "trailing_bytes" end
	if path>math.max(maxx-minx,maxy-miny,1)*P.MAX_INK_SPANS then return nil,"ink_budget" end
	out.points = total
	return out
end

return table.freeze(P)
