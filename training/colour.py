"""Ink colours: the palette, which colours each thing is usually drawn in, and how
training drawings get coloured.

Quick, Draw! and the synthetic doodles are black ink only. Until real coloured
player drawings exist, training colours them here from PRIORS (Claude's guesses of
what people pick: an apple red or green with a brown stem, a bomb black with an
orange spark). The network therefore learns roughly what this table says; real
player drawings replace it later.

Colour must never be required: COLOURISE keeps a large share of drawings all black
(the default ink) and some in random colours, so an uncoloured or oddly coloured
drawing still reads by shape.

Network input (2026-11-09: colour goes into the network): 4 planes per pixel,
    ink, ink*r, ink*g, ink*b
with (r, g, b) = PALETTE[i].net for the stroke that owns the pixel. Black has
net = (0, 0, 0), so a black drawing is exactly today's single ink plane plus three
zero planes. luau/DrawPalette.luau holds the same palette for the game.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ink:
    name: str
    rgb: tuple  # what the player sees on the cream paper (246, 241, 230)
    net: tuple  # what the network gets, 0..1; black is all zero


PALETTE = [
    Ink("black", (28, 28, 32), (0.0, 0.0, 0.0)),
    Ink("gray", (128, 132, 140), (0.50, 0.52, 0.55)),
    Ink("brown", (130, 82, 45), (0.51, 0.32, 0.18)),
    Ink("red", (226, 48, 48), (0.89, 0.19, 0.19)),
    Ink("orange", (245, 140, 30), (0.96, 0.55, 0.12)),
    Ink("yellow", (245, 205, 30), (0.96, 0.80, 0.12)),
    Ink("green", (60, 170, 70), (0.24, 0.67, 0.27)),
    Ink("blue", (45, 110, 225), (0.18, 0.43, 0.88)),
    Ink("purple", (140, 70, 200), (0.55, 0.27, 0.78)),
    Ink("pink", (240, 120, 180), (0.94, 0.47, 0.71)),
]
INDEX = {c.name: i for i, c in enumerate(PALETTE)}
NET = np.array([c.net for c in PALETTE], np.float32)  # (10, 3)

# "main | accent". main colours the big strokes (body, outline), accent the small
# ones (stem, spark, eyes, details). "name3" = weight 3, no number = 1.
# "~a b c" gives every stroke the next colour of the list (rainbow, traffic light).
# "any" = every palette colour equally (shapes and symbols have no real colour).
PRIORS = {
    "anvil": "gray3 black2 | gray black",
    "piano": "black3 brown2 | black gray",
    "hammer": "brown3 gray2 | gray2 brown",
    "axe": "brown3 gray2 | gray3",
    "sword": "gray4 blue | brown2 yellow2 gray",
    "knife": "gray4 | brown2 black",
    "saw": "gray3 | brown2 red",
    "baseball bat": "brown4 | brown black",
    "frying pan": "black3 gray2 | black brown",
    "boomerang": "brown3 orange red | brown",
    "rifle": "black3 brown2 gray | black brown",
    "cannon": "black3 gray2 | brown3 black",
    "tornado": "gray4 black2 | gray brown",
    "hurricane": "gray3 blue2 | gray blue",
    "lightning": "yellow5 orange blue | yellow",
    "rain": "blue3 gray2 | blue3 gray",
    "campfire": "orange3 red2 yellow | brown4 yellow red",
    "shark": "gray4 blue2 | black red",
    "crocodile": "green5 | green black yellow",
    "dragon": "green3 red3 purple | orange2 red yellow",
    "snake": "green4 brown yellow red | red2 black",
    "spider": "black5 | black red",
    "scorpion": "black3 brown2 orange | black red",
    "bee": "yellow4 black | black3 blue",
    "lion": "orange3 yellow2 brown2 | brown2 orange black",
    "bulldozer": "yellow5 orange | black2 gray2",
    "helicopter": "gray2 green2 blue red black | black2 gray",
    "airplane": "gray3 blue2 | blue gray red",
    "submarine": "yellow3 gray2 black | gray blue",
    "police car": "blue3 black2 | blue red2 black",
    "firetruck": "red5 | gray2 black yellow",
    "flying saucer": "gray3 green | green2 yellow2 blue",
    "shovel": "brown2 gray2 | gray2 brown2",
    "drill": "yellow2 orange gray2 black | gray2 black",
    "scissors": "gray3 red blue | red2 black gray",
    "fork": "gray5 | gray",
    "rake": "brown3 gray | gray2 brown",
    "screwdriver": "red2 yellow2 blue black | gray3",
    "pliers": "red2 blue gray2 black | gray3",
    "golf club": "gray3 black | black2 gray",
    "hockey stick": "brown3 black | black2",
    "dumbbell": "gray3 black3 | gray2 black",
    "bear": "brown5 black | black2 brown",
    "tiger": "orange5 | black4",
    "rhinoceros": "gray5 | gray black",
    "elephant": "gray5 | gray black",
    "octopus": "purple3 red2 pink2 orange | purple black",
    "whale": "blue4 gray2 | blue2 black",
    "bat": "black4 purple gray | black red",
    "ant": "black3 red2 brown | black",
    "mosquito": "black2 gray2 brown | black red",
    "crab": "red4 orange2 | black2 red",
    "lobster": "red5 | black2 red",
    "hedgehog": "brown4 gray | brown black2",
    "tractor": "green3 red2 yellow | black3 gray",
    "train": "red2 blue black2 gray green | black2 gray",
    "truck": "red2 blue2 gray2 yellow | black3 gray",
    "aircraft carrier": "gray5 | gray blue black",
    "speedboat": "red2 blue2 orange | blue2 gray",
    "cactus": "green5 | green pink yellow",
    "tree": "green4 brown2 | brown3 green2 red",
    "mountain": "gray3 brown2 green | blue gray",
    "snowflake": "blue5 | blue",
    "sun": "yellow5 orange2 | orange2 yellow2",
    "cloud": "gray3 blue3 black | gray blue",
    "moon": "yellow4 gray2 | gray2 yellow",
    "star": "yellow5 orange | yellow orange",
    "rainbow": "~red orange yellow green blue purple | blue gray",
    "ocean": "blue5 | blue2 gray",
    "snowman": "gray2 blue2 black | orange2 black2 brown",
    "ice cream": "pink3 brown2 yellow | brown3 orange red",
    "fire hydrant": "red5 yellow | gray red",
    "bucket": "gray2 blue2 red yellow | gray black",
    "umbrella": "red2 blue2 purple yellow black | black2 gray brown",
    "fan": "gray3 blue | gray2 black",
    "hourglass": "brown2 yellow gray | yellow3 orange",
    "alarm clock": "red2 black2 gray | black2 red",
    "light bulb": "yellow4 gray | gray3 yellow",
    "stop sign": "red5 | gray2 black",
    "jail": "gray4 black2 | gray brown",
    "hot air balloon": "red2 orange yellow blue purple | brown3 black",
    "megaphone": "red2 gray yellow black | black2 gray",
    "drums": "red2 blue brown2 | brown2 gray2 yellow",
    "bandage": "orange brown2 pink | brown pink red",
    "foot": "orange pink2 brown | pink brown",
    "hand": "orange pink2 brown | pink brown",
    "skull": "gray3 black3 | black3",
    "car": "red3 blue2 gray | black3 gray",
    "house": "brown2 red2 orange | red2 brown2 blue yellow",
    "castle": "gray5 | brown2 red blue",
    "The Eiffel Tower": "gray3 brown2 black | gray",
    "toilet": "gray3 blue | gray blue",
    "banana": "yellow6 | brown2 green",
    "pizza": "yellow3 orange2 | red4 brown green",
    "mushroom": "red4 brown2 | brown2 gray yellow",
    "cow": "black2 brown2 gray | black3 pink2",
    "pig": "pink6 | pink black",
    "frog": "green6 | black red",
    "penguin": "black5 | orange3 yellow",
    "soccer ball": "black4 gray | black3",
    "candle": "red2 pink yellow blue | orange3 yellow2",
    "fireplace": "brown2 red2 gray2 | orange3 red2 yellow",
    "lantern": "yellow2 brown red gray | orange2 yellow2",
    "matches": "brown3 | red4 orange",
    "lighter": "red2 blue2 gray black | orange3 red",
    "river": "blue6 | blue green",
    "flower": "red2 pink2 purple yellow | green5",
    "palm tree": "brown3 green2 | green4",
    "leaf": "green5 orange red yellow | brown green",
    "bush": "green6 | green red",
    "bathtub": "gray3 blue | blue3 gray",
    "bed": "brown3 blue red | blue red brown pink",
    "couch": "red2 brown2 blue gray | brown gray",
    "chair": "brown5 | brown2 gray",
    "table": "brown5 | brown",
    "bench": "brown4 green | gray2 brown black",
    "dresser": "brown5 | gray yellow",
    "washing machine": "gray4 | blue2 gray",
    "microwave": "gray3 black2 | black2 gray",
    "oven": "gray3 black2 | black red",
    "stove": "gray3 black2 | red2 black blue",
    "television": "black4 gray2 | blue2 gray",
    "computer": "gray3 black2 | blue2 black",
    "laptop": "gray3 black2 | blue2 black",
    "toaster": "gray4 red | brown2 black",
    "wheel": "black5 gray | gray2 black",
    "ladder": "brown4 gray2 | brown gray",
    "door": "brown5 | yellow2 gray",
    "fence": "brown4 gray | brown",
    "mailbox": "blue2 gray red black | red2 brown2",
    "streetlight": "gray3 black2 | yellow4",
    "traffic light": "black3 gray yellow | ~red yellow green",
    "cooler": "blue2 red2 gray | gray2 blue",
    "baseball": "gray3 black | red5",
    "basketball": "orange5 | black4",
    "tennis racquet": "black2 red blue gray | gray yellow green",
    "hockey puck": "black6 | gray",
    "broom": "brown3 yellow | yellow3 brown",
    "nail": "gray5 | gray",
    "syringe": "gray2 blue | red2 gray blue",
    "key": "yellow3 gray3 | yellow gray",
    "helmet": "red2 blue gray2 black yellow | black gray",
    "crown": "yellow5 orange | red2 blue2 green purple",
    "diamond": "blue4 purple gray | blue gray",
    "dog": "brown4 black gray orange | black2 brown pink",
    "cat": "orange2 black2 gray2 brown | black2 pink2",
    "horse": "brown5 black | black2 brown",
    "bird": "blue2 red yellow brown black | orange3 yellow black",
    "owl": "brown5 gray | orange2 yellow2 black",
    "duck": "yellow4 brown | orange4",
    "parrot": "green3 red2 blue | yellow2 black blue",
    "flamingo": "pink6 | black2 pink",
    "swan": "gray3 blue black | orange3 black",
    "fish": "orange2 blue2 yellow green | black orange",
    "dolphin": "blue3 gray3 | gray black",
    "sea turtle": "green5 | brown2 green",
    "snail": "brown3 green yellow | gray2 brown",
    "butterfly": "blue2 purple orange yellow pink | black3",
    "giraffe": "yellow3 orange2 | brown4",
    "zebra": "black3 gray | black4",
    "camel": "brown3 orange2 yellow | brown black",
    "kangaroo": "brown4 orange | brown black",
    "monkey": "brown5 | orange brown pink",
    "panda": "black4 gray | black3",
    "rabbit": "gray3 brown2 pink | pink3 black",
    "raccoon": "gray4 black | black3 gray",
    "sheep": "gray3 black | black3",
    "squirrel": "brown3 orange2 | brown black",
    "mouse": "gray5 | pink3 black",
    "teddy-bear": "brown5 | black brown red",
    "mermaid": "green2 blue orange pink | red2 yellow2 green",
    "angel": "yellow2 gray blue | yellow3 gray",
    "bus": "yellow2 red blue | black3 gray",
    "school bus": "yellow5 orange | black3",
    "ambulance": "gray3 red | red4",
    "van": "gray2 blue2 red | black3",
    "pickup truck": "red2 blue2 black | black3 gray",
    "motorbike": "black3 red2 gray | black2 gray",
    "bicycle": "black3 gray red blue | black2",
    "skateboard": "black2 red blue green brown | black2 gray",
    "sailboat": "brown3 blue | red2 blue yellow gray",
    "canoe": "brown3 red green | brown blue",
    "cruise ship": "gray3 blue2 | blue2 red black",
    "parachute": "red2 orange blue green | black2 gray",
    "roller coaster": "red2 blue brown gray | gray2 red",
    "barn": "red5 | brown2 gray",
    "church": "gray2 brown2 | yellow2 brown",
    "hospital": "gray3 | red4 blue",
    "lighthouse": "red3 gray | yellow3 red",
    "skyscraper": "gray4 blue2 black | blue2 yellow",
    "windmill": "brown3 gray red | gray brown",
    "bridge": "gray3 brown2 red | blue3 gray",
    "tent": "green2 orange blue brown red | brown2 black",
    "The Great Wall of China": "gray3 brown2 | green gray",
    "apple": "red5 green3 yellow | brown3 green3",
    "watermelon": "green4 | red4 black2",
    "pineapple": "yellow4 orange | green5",
    "hamburger": "brown3 orange2 yellow | green2 red2 yellow",
    "hot dog": "brown2 orange red | red2 yellow2",
    "donut": "brown2 pink2 orange | ~pink red yellow blue",
    "cake": "brown2 pink2 yellow | red2 pink yellow",
    "birthday cake": "pink2 brown2 yellow | yellow2 orange2 red blue",
    "cookie": "brown5 orange | brown2 black",
    "bread": "brown3 orange2 yellow | brown2",
    "steak": "brown3 red2 | brown gray",
    "sandwich": "brown2 yellow2 orange | green2 red2 yellow",
    "strawberry": "red6 | green4 yellow",
    "grapes": "purple5 green2 | green3 brown",
    "carrot": "orange6 | green5",
    "broccoli": "green6 | green2",
    "lollipop": "~red pink yellow blue purple orange | gray2 brown",
    "popsicle": "red2 orange pink blue purple | brown3",
    "potato": "brown4 yellow | brown2",
    "onion": "purple2 brown2 yellow | brown green",
    "pear": "green3 yellow3 | brown3 green",
    "cup": "blue red gray yellow | brown gray",
    "coffee cup": "gray2 brown2 red | brown2 gray",
    "mug": "red blue gray brown | brown2 gray",
    "teapot": "gray2 blue red brown | gray brown",
    "wine bottle": "green3 brown red | red2 gray",
    "wine glass": "gray3 blue | red4 purple",
    "book": "red2 blue2 brown green | black2 gray",
    "pencil": "yellow4 orange | pink2 black2 gray",
    "clock": "gray2 black2 brown | black3 red",
    "wristwatch": "black2 gray2 brown | black2 gray",
    "hat": "black2 brown2 red blue | red black",
    "shoe": "brown2 black2 red blue | gray black",
    "sock": "red blue gray green | red gray",
    "eyeglasses": "black4 brown | blue2 gray",
    "headphones": "black4 gray red | gray black",
    "guitar": "brown4 red orange | black2 brown",
    "violin": "brown5 orange | black2",
    "trumpet": "yellow5 orange | yellow gray",
    "saxophone": "yellow5 orange | yellow gray",
    "microphone": "gray3 black3 | black2",
    "cell phone": "black4 gray2 | blue2 gray",
    "camera": "black3 gray2 | gray2 blue",
    "radio": "brown2 gray2 black red | gray2 black",
    "backpack": "red2 blue2 green black | gray black",
    "suitcase": "brown3 black2 gray | gray brown",
    "envelope": "gray2 brown yellow | red2 gray",
    "pillow": "gray2 blue pink | gray",
    "toothbrush": "blue2 red green | blue2 gray",
    "vase": "blue2 brown red | red2 pink yellow green",
    "house plant": "brown3 orange | green5",
    "flashlight": "gray2 black2 yellow red | yellow4",
    "binoculars": "black4 gray | blue gray",
    "compass": "gray2 brown yellow | red3 black",
    "t-shirt": "red2 blue2 green yellow | black gray",
    "pants": "blue5 brown black | blue brown",
    "smiley face": "yellow5 | black4",
    "face": "orange pink2 brown | black3 red",
    "eye": "black2 gray | blue3 green brown",
    "brain": "pink5 | pink2 red",
    "tooth": "gray3 yellow | gray",
    "mouth": "red4 pink2 | gray red",
    "nose": "pink2 orange brown | brown black",
    "ear": "pink2 orange brown | brown pink",
    "moustache": "black3 brown3 | black",
    "beard": "brown3 black2 orange | brown black",
    "arm": "pink2 orange brown | pink brown",
    "leg": "pink orange brown blue | brown black",
    "circle": "any | any",
    "line": "any | any",
    "square": "any | any",
    "triangle": "any | any",
    "hexagon": "any | any",
    "octagon": "any | any",
    "squiggle": "any | any",
    "zigzag": "any | any",
    # the 29 synthetic things
    "bomb": "black6 gray | orange3 yellow2 red2 brown",
    "nuke": "yellow3 black2 gray green | black3 orange red",
    "meteor": "brown2 gray2 orange | orange3 red2 yellow2",
    "volcano": "brown3 gray2 | red3 orange3 yellow",
    "tsunami": "blue6 | blue2 gray",
    "black hole": "black5 purple2 | purple2 blue orange yellow",
    "pistol": "black4 gray2 | brown2 black",
    "tank": "green5 gray brown | black2 green",
    "rocket": "gray2 red2 blue | orange3 red2 yellow2",
    "grenade": "green5 gray | gray2 black",
    "dynamite": "red6 | brown2 yellow2 orange2",
    "ninja star": "gray5 black | gray",
    "bow and arrow": "brown4 | gray2 brown red",
    "shield": "gray2 blue2 red brown yellow | yellow2 red2 gray",
    "dinosaur": "green5 brown | green black",
    "zombie": "green5 gray | red2 black brown",
    "ghost": "gray3 blue purple black | black3",
    "robot": "gray5 blue | red2 blue yellow",
    "alien": "green6 | black3 green",
    "poop": "brown7 | brown black",
    "heart": "red6 pink3 | red pink",
    "taco": "yellow3 orange2 | green3 red2 brown",
    "cheese": "yellow5 orange2 | yellow orange",
    "straw hat": "yellow4 orange | red5",
    "kunai": "gray4 black | brown2 gray red",
    "spiral": "any | any",
    "cursed face": "pink orange gray | black3 red2 purple",
    "x2": "any | any",
    "undo arrow": "any | any",
}

# Shares of the training colourings. Black and random keep colour optional.
P_BLACK = 0.40    # all strokes black, as today
P_RANDOM = 0.10   # one or two random colours, unrelated to the thing
# the remaining 0.50 follow PRIORS, split into these styles:
S_SINGLE = 0.35   # the whole drawing in its main colour
S_OUTLINE = 0.20  # big strokes black, details in colour (outline first, then colour)
#                   rest: big strokes main colour, small strokes accent colour
BIG = 0.20        # a stroke with at least this share of the ink counts as big


def _parse(part):
    part = part.strip()
    if part == "any":
        return ("w", np.ones(len(PALETTE)) / len(PALETTE))
    if part.startswith("~"):
        return ("cycle", [INDEX[n] for n in part[1:].split()])
    w = np.zeros(len(PALETTE))
    for tok in part.split():
        name = tok.rstrip("0123456789")
        w[INDEX[name]] += float(tok[len(name):] or 1)
    return ("w", w / w.sum())


_PARSED = {k: tuple(_parse(p) for p in v.split("|")) for k, v in PRIORS.items()}


def stroke_ink(strokes):
    """Ink length of each stroke, from (xs, ys) pairs in any coordinate space."""
    out = []
    for xs, ys in strokes:
        x, y = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        out.append(float(np.hypot(np.diff(x), np.diff(y)).sum()) + 1.0)
    return out


def _pick(spec, rng, n):
    kind, v = spec
    if kind == "cycle":
        k = int(rng.integers(len(v)))
        return [v[(k + i) % len(v)] for i in range(n)]
    c = int(rng.choice(len(PALETTE), p=v))
    return [c] * n


def colourise(cls, ink, rng):
    """Palette index for each stroke of one drawing of `cls`.

    ink: per-stroke ink lengths (stroke_ink). rng: numpy Generator.
    Call it again every epoch so the same drawing is seen in different colourings.
    """
    n = len(ink)
    tot = sum(ink) or 1.0
    big = [s / tot >= BIG for s in ink]
    big[int(np.argmax(ink))] = True
    r = rng.random()
    if r < P_BLACK:
        return [0] * n
    if r < P_BLACK + P_RANDOM:
        a, b = (int(c) for c in rng.integers(len(PALETTE), size=2))
        if rng.random() < 0.5:
            return [a] * n
        return [a if g else b for g in big]
    main_spec, accent_spec = _PARSED[cls]
    main = _pick(main_spec, rng, n)
    accent = _pick(accent_spec, rng, n)
    s = rng.random()
    if s < S_SINGLE:
        return main
    if s < S_SINGLE + S_OUTLINE:
        return [0 if g else accent[i] for i, g in enumerate(big)]
    return [main[i] if g or rng.random() < 0.25 else accent[i] for i, g in enumerate(big)]


def planes(ink_plane, owner, colours):
    """4-plane network input from a rendered drawing.

    ink_plane: (H, W) float ink 0..1. owner: (H, W) int index of the stroke that
    owns each pixel (the nearest one). colours: palette index per stroke.
    """
    rgb = NET[np.asarray(colours, np.int64)[owner]]  # (H, W, 3)
    return np.concatenate([ink_plane[None], (ink_plane[..., None] * rgb).transpose(2, 0, 1)], 0)


def _check():
    from categories import CATEGORIES
    missing = [c for c in CATEGORIES if c not in PRIORS]
    extra = [c for c in PRIORS if c not in CATEGORIES]
    assert not missing and not extra, (missing, extra)
    rng = np.random.default_rng(0)
    for c in CATEGORIES:
        colourise(c, [10.0, 3.0, 1.0], rng)
    print(f"colour priors ok: {len(PRIORS)} things, {len(PALETTE)} inks")


if __name__ == "__main__":
    _check()
