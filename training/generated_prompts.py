"""Hand-written semantic descriptions; no test-set pictures or captions used."""
from categories import CATEGORIES

DESCRIPTIONS={
 'bomb':'a round cartoon bomb with a fuse and a small spark',
 'nuke':'a nuclear explosion with a mushroom cloud',
 'meteor':'a meteor rock with a long fiery tail',
 'volcano':'an erupting volcano', 'tsunami':'a large curling tsunami wave',
 'black hole':'a black hole in space, a dark central circle with an elliptical accretion disk',
 'pistol':'a handgun pistol', 'tank':'an army tank with caterpillar tracks and a cannon',
 'rocket':'a space rocket', 'grenade':'a hand grenade with a safety pin',
 'dynamite':'sticks of dynamite with a fuse', 'ninja star':'a shuriken ninja throwing star',
 'bow and arrow':'an archery bow with its arrow', 'shield':'a medieval shield',
 'dinosaur':'a dinosaur', 'zombie':'a cartoon zombie with damaged skin and missing teeth',
 'ghost':'a sheet ghost', 'robot':'a robot', 'alien':'an alien with a large head and large eyes',
 'poop':'a cartoon pile of poop', 'heart':'a heart symbol', 'taco':'a taco with a folded shell',
 'cheese':'a wedge of cheese with holes', 'straw hat':'a woven straw sun hat with a wide brim',
 'kunai':'a ninja kunai throwing knife with a triangular blade and a ring handle',
 'spiral':'a simple spiral line', 'cursed face':'a creepy cursed face with angry eyes and a scary grin',
 'x2':'the handwritten multiplier symbol x2', 'undo arrow':'a curved undo arrow pointing back to the left',
}
STYLES={
 'marker':'a quick rough black marker doodle, few simple uneven pen strokes',
 'outline':'a careful hand-drawn outline sketch, simple clear shapes, black pen lines',
 'flat':'a simple hand-drawn shape filled with flat colours, bold irregular edges',
 'painted':'a rough hand-painted drawing with broad visible brush strokes and uneven filled areas',
 'kids':"a child's simple crayon drawing, imperfect proportions, bright flat colours and scribbled fills",
}
VIEWS=['side view','front view','slightly tilted view','simple three-quarter view','a simple iconic view']
EXTRAS=['very simple, only essential parts','a few identifying details','loose wonky freehand lines','a clear silhouette with minimal detail']


def description(name):return DESCRIPTIONS.get(name,name)


def prompt(name,style,attempt):
    desc=description(name)
    variants={
        'pistol':['a handgun pistol','a revolver handgun with a cylinder','a compact semi-automatic pistol','a simple blocky handgun'],
        'grenade':['a segmented hand grenade with a safety pin','a smooth round hand grenade with a pull ring','a stick hand grenade','an egg-shaped hand grenade with a lever'],
        'dinosaur':['a friendly long-neck dinosaur','a stegosaurus dinosaur','a cartoon tyrannosaurus dinosaur','a small bipedal dinosaur'],
        'robot':['a simple boxy robot with a very large square head and two big widely spaced eyes','a robot head with antennae and two widely spaced round eyes','a round friendly toy robot with short legs','a simple robot bust with a large head and block-shaped arms'],
        'zombie':['a simple green zombie face with big uneven eyes and a few missing teeth','a zombie head with one scar and ragged hair','a simple full-body zombie with outstretched arms and a very large head','a cartoon zombie bust with a torn shirt and a large head'],
        'dynamite':['a single red stick of dynamite with a long curled lit fuse','three red sticks of dynamite tied together with a lit fuse','a simple bundle of dynamite with a large arching fuse','two red sticks of dynamite with a black band and one sparking fuse'],
        'poop':['a simple brown poop emoji with three stacked curved tiers','a cartoon brown swirl pile of poop with big white eyes','a simple pointed brown pile of poop with three broad curved tiers','a brown cartoon poop swirl with a rounded base'],
        'cursed face':['a scary cursed face with two huge crooked eyes and a broad menacing grin','a simple creepy face with big wide eyes and a sharp grin','an unsettling cursed face with wide circular eyes and a crooked grin','a scary cursed face drawn with only two big eyes and a crooked mouth'],
        'straw hat':['a woven straw sun hat with a very wide brim and a hatband, only a few large weave checks','a wide-brim straw hat with sparse large woven lines','a simple round straw sun hat with a broad brim and ribbon','a straw hat with a wide oval brim and a few clear crosshatch marks'],
        'cheese':['a wedge of cheese with holes','a rectangular slice of Swiss cheese with holes','a thick triangle of cheese','a block of cheese with round holes'],
    }
    if name in variants:desc=variants[name][(attempt//25)%len(variants[name])]
    view=VIEWS[(attempt//len(STYLES))%len(VIEWS)]
    if name in ['heart','spiral','x2','undo arrow','nuke','black hole']:view='simple flat view'
    if name in ['cursed face','poop']:view='simple front view'
    if name in ['robot','zombie'] and ('face' in desc or 'head with' in desc):view='simple front view'
    detail=EXTRAS[(attempt//(len(STYLES)*len(VIEWS)))%len(EXTRAS)]
    avoid='no unrelated text' if name=='x2' else 'no labels'
    return (f'On plain pure white paper, one isolated drawing of {desc}. '
            f'{STYLES[style]}. {view}, {detail}. '
            'The whole drawing fits with a wide white margin. No scene, no background objects, '
            f'no frame, no signature, {avoid}, no watermark. Hand-drawn doodle, not a photograph.')


def teacher_texts():
    texts=[]
    for name in CATEGORIES:
        texts += [f'a drawing of {name}',f'a simple doodle of {description(name)}',
                  f'a coloured illustration of {name}',f'a black and white sketch of {description(name)}']
    negatives=['a blank white sheet of paper','random unrecognisable scribbles','a page of printed words',
               'an abstract texture or pattern','a complicated scene containing many objects','a photograph of a landscape']
    return texts,negatives
