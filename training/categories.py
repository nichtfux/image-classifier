# Append-only recognition vocabulary: 270 Quick, Draw! classes, then 29 custom classes.
# The first 270 match Google's categories; the final 29 use hand-reviewed synthetic strokes.
CATEGORIES = [
    # things that drop or hit
    "anvil", "piano", "hammer", "axe", "sword", "knife", "saw", "baseball bat", "frying pan", "boomerang",
    # guns and launchers
    "rifle", "cannon",
    # weather
    "tornado", "hurricane", "lightning", "rain", "campfire",
    # creatures
    "shark", "crocodile", "dragon", "snake", "spider", "scorpion", "bee", "lion",
    # vehicles
    "bulldozer", "helicopter", "airplane", "submarine", "police car", "firetruck", "flying saucer",
    # more tools and heavy objects (append-only: the first 32 labels stay stable)
    "shovel", "drill", "scissors", "fork", "rake", "screwdriver", "pliers", "golf club", "hockey stick", "dumbbell",
    # more creatures
    "bear", "tiger", "rhinoceros", "elephant", "octopus", "whale", "bat", "ant", "mosquito", "crab", "lobster", "hedgehog",
    # more vehicles
    "tractor", "train", "truck", "aircraft carrier", "speedboat",
    # nature powers and things to drop
    "cactus", "tree", "mountain", "snowflake", "sun",
    # 100-category round (2026-11-09), still append-only.
    # sky and weather, cold things
    "cloud", "moon", "star", "rainbow", "ocean", "snowman", "ice cream",
    # water, gadgets and traps
    "fire hydrant", "bucket", "umbrella", "fan", "hourglass", "alarm clock", "light bulb", "stop sign", "jail",
    "hot air balloon", "megaphone", "drums", "bandage",
    # stomp, slap, curse
    "foot", "hand", "skull",
    # heavy things to drop
    "car", "house", "castle", "The Eiffel Tower", "toilet",
    # food
    "banana", "pizza", "mushroom",
    # animals
    "cow", "pig", "frog", "penguin",
    # balls
    "soccer ball",
    # 270-category round: exact order from notes/category-plan.md.
    # Fire and nature (10)
    "candle", "fireplace", "lantern", "matches", "lighter", "river",
    "flower", "palm tree", "leaf", "bush",
    # Heavy things to drop: furniture, appliances, street things (23)
    "bathtub", "bed", "couch", "chair", "table", "bench",
    "dresser", "washing machine", "microwave", "oven", "stove", "television",
    "computer", "laptop", "toaster", "wheel", "ladder", "door",
    "fence", "mailbox", "streetlight", "traffic light", "cooler",
    # Sport, weapons-ish and power-ups (11)
    "baseball", "basketball", "tennis racquet", "hockey puck", "broom", "nail",
    "syringe", "key", "helmet", "crown", "diamond",
    # Animals and creatures to summon (28)
    "dog", "cat", "horse", "bird", "owl", "duck",
    "parrot", "flamingo", "swan", "fish", "dolphin", "sea turtle",
    "snail", "butterfly", "giraffe", "zebra", "camel", "kangaroo",
    "monkey", "panda", "rabbit", "raccoon", "sheep", "squirrel",
    "mouse", "teddy-bear", "mermaid", "angel",
    # Vehicles (13)
    "bus", "school bus", "ambulance", "van", "pickup truck", "motorbike",
    "bicycle", "skateboard", "sailboat", "canoe", "cruise ship", "parachute",
    "roller coaster",
    # Buildings and landmarks (9)
    "barn", "church", "hospital", "lighthouse", "skyscraper", "windmill",
    "bridge", "tent", "The Great Wall of China",
    # Food: funny things to drop (21)
    "apple", "watermelon", "pineapple", "hamburger", "hot dog", "donut",
    "cake", "birthday cake", "cookie", "bread", "steak", "sandwich",
    "strawberry", "grapes", "carrot", "broccoli", "lollipop", "popsicle",
    "potato", "onion", "pear",
    # Everyday objects (35)
    "cup", "coffee cup", "mug", "teapot", "wine bottle", "wine glass",
    "book", "pencil", "clock", "wristwatch", "hat", "shoe",
    "sock", "eyeglasses", "headphones", "guitar", "violin", "trumpet",
    "saxophone", "microphone", "cell phone", "camera", "radio", "backpack",
    "suitcase", "envelope", "pillow", "toothbrush", "vase", "house plant",
    "flashlight", "binoculars", "compass", "t-shirt", "pants",
    # Body and faces (12)
    "smiley face", "face", "eye", "brain", "tooth", "mouth",
    "nose", "ear", "moustache", "beard", "arm", "leg",
    # Scribble classes, no game effect (8)
    "circle", "line", "square", "triangle", "hexagon", "octagon",
    "squiggle", "zigzag",
    # hand-reviewed synthetic classes (2026-11-09), exact agreed order.
    "bomb", "nuke", "meteor", "volcano", "tsunami", "black hole", "pistol",
    "tank", "rocket", "grenade", "dynamite", "ninja star", "bow and arrow",
    "shield", "dinosaur", "zombie", "ghost", "robot", "alien", "poop", "heart",
    "taco", "cheese", "straw hat", "kunai", "spiral", "cursed face", "x2", "undo arrow",
]
assert len(CATEGORIES) == len(set(CATEGORIES))

# Raw Google download/render tools must never request the custom class names.
REAL_CATEGORIES = CATEGORIES[:270]
SYNTH_CATEGORIES = CATEGORIES[270:]
