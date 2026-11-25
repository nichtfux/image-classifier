--!strict
-- DrawPalette: the ink colours players can draw with. Same order and values as
-- training/colour.py PALETTE; keep the two in sync. Index 1 (black) is the default ink.
-- `net` is what the recogniser gets per ink pixel (ink*r, ink*g, ink*b); black is
-- all zero, so a black drawing reads exactly like the old single-plane input.
export type Ink = { name: string, color: Color3, net: Vector3 }

local function ink(name: string, r: number, g: number, b: number, net: Vector3): Ink
	return { name = name, color = Color3.fromRGB(r, g, b), net = net }
end

local Palette: { Ink } = {
	ink("black", 28, 28, 32, Vector3.new(0, 0, 0)),
	ink("gray", 128, 132, 140, Vector3.new(0.50, 0.52, 0.55)),
	ink("brown", 130, 82, 45, Vector3.new(0.51, 0.32, 0.18)),
	ink("red", 226, 48, 48, Vector3.new(0.89, 0.19, 0.19)),
	ink("orange", 245, 140, 30, Vector3.new(0.96, 0.55, 0.12)),
	ink("yellow", 245, 205, 30, Vector3.new(0.96, 0.80, 0.12)),
	ink("green", 60, 170, 70, Vector3.new(0.24, 0.67, 0.27)),
	ink("blue", 45, 110, 225, Vector3.new(0.18, 0.43, 0.88)),
	ink("purple", 140, 70, 200, Vector3.new(0.55, 0.27, 0.78)),
	ink("pink", 240, 120, 180, Vector3.new(0.94, 0.47, 0.71)),
}

return table.freeze(Palette)
