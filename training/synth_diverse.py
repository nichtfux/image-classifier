"""Independent hand-written drawing families, not images copied from test sets.

Original reviewed programs remain unchanged. New seeds have their own train,
validation and test namespaces. These tests are still procedural upper bounds.
"""
import argparse, json, math, os, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_dilation
from synth_doodles import P, arc, ell, rect, qbez, bez, star_poly, humanise, to_quickdraw, THINGS, TAU
from colour_raster import render
from colour import colourise, planes, NET
from categories import CATEGORIES

ROOT=Path(__file__).resolve().parent.parent/'data/synth-diverse40'


def circle(x,y,r,rng):return ell(x,y,r,r*rng.uniform(.86,1.14),rng)
def line(*p):return P(*p)


def family(name,rng,view):
    """Small symbol, detailed cartoon and different viewpoint families."""
    e=lambda x,y,a,b:ell(x,y,a,b,rng)
    if name=='bomb':
        cx,cy=50,62;r=rng.uniform(23,33)
        s=[circle(cx,cy,r,rng)]
        a=-math.pi/2 if view==0 else rng.uniform(-2.6,-.3)
        ux,uy=np.cos(a),np.sin(a);vx,vy=-uy,ux
        cap=np.array([cx+ux*r,cy+uy*r]);w=rng.uniform(4,9);height=rng.uniform(2,7)
        s+=[P(cap+np.array([vx,vy])*w,cap+np.array([vx,vy])*w+np.array([ux,uy])*height,
              cap-np.array([vx,vy])*w+np.array([ux,uy])*height,cap-np.array([vx,vy])*w)]
        start=cap+np.array([ux,uy])*height;length=rng.uniform(8,27)
        end=start+np.array([ux,uy])*length
        s+=[qbez(start,start+np.array([vx,vy])*rng.uniform(-9,9)+np.array([ux,uy])*length*.6,end)]
        if view!=0 or rng.random()<.5:
            for a in np.linspace(0,TAU,int(rng.integers(4,9)),endpoint=False):
                d=np.array([np.cos(a),np.sin(a)]);s+=[P(end+d*3,end+d*rng.uniform(6,11))]
        if view==2:s+=[arc(cx,cy,r*.65,r*.65,3.3,4.6)]
        return s
    if name=='pistol':
        if view==0:
            s=[line((12,25),(84,25),(84,40),(51,40),(46,77),(29,77),(35,40),(12,40),(12,25)),
               arc(51,43,14,14,0,math.pi),line((52,40),(52,47))]
        elif view==1:
            # Revolver: cylinder, short barrel and sloping grip, rather than one L silhouette.
            s=[rect(19,30,28,22),rect(47,32,41,12),line((22,52),(14,81),(31,87),(41,53)),
               e(34,41,10,9),arc(44,52,11,12,-.2,2.7),line((47,51),(46,57))]
            if rng.random()<.65:s += [e(31,37,2,2),e(37,43,2,2),line((82,30),(82,26))]
        else:
            s=[line((9,31),(90,31),(90,44),(56,44),(50,83),(30,78),(36,44),(9,44),(9,31)),
               line((14,36),(81,36)),arc(53,49,12,10,0,math.pi),line((60,44),(55,54))]
        return s
    if name=='grenade':
        if view==0:
            s=[e(47,60,23,30),rect(40,24,15,10),e(64,24,11,10),line((54,30),(65,47),(64,65))]
        elif view==1:
            s=[line((36,35),(60,35),(73,52),(70,79),(57,89),(35,85),(23,67),(25,48),(36,35)),
               rect(37,25,20,10),e(64,24,10,9),line((52,27),(70,34),(79,58))]
        else:s=[e(45,52,23,24),rect(39,73,12,24),rect(37,22,18,8),e(65,23,10,9)]
        if view!=2:
            for y in [48,61,73]:s+=[line((29,y),(65,y))]
            if view==1:s+=[line((40,39),(37,82)),line((52,37),(56,83))]
        return s
    if name=='black hole':
        # Never a plain spiral or a generic isolated filled circle.
        s=[e(50,51,rng.uniform(9,15),rng.uniform(9,14))]
        for dy in np.arange(-8,10,2.5):s += [line((41,51+dy),(59,51+dy))]
        if view==0:
            s += [e(50,51,43,11),e(50,51,35,17)]
        elif view==1:
            s += [arc(50,51,44,17,math.pi*.04,math.pi*.95),arc(50,51,44,17,math.pi*1.05,math.pi*1.96),
                  bez((7,44),(20,70),(78,14),(93,53)),bez((9,60),(26,21),(72,82),(94,46))]
        else:
            s += [e(50,51,40,16),qbez((12,30),(70,0),(84,43)),qbez((88,74),(20,95),(15,55))]
        a=rng.uniform(-.7,.7);m=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        return [(q-np.array([50,50]))@m.T+50 for q in s]
    if name=='tank':
        if view==0:return [line((12,58),(81,58),(90,75),(81,88),(15,88),(6,77),(12,58)),rect(25,38,42,20),rect(57,44,38,6)]+[circle(x,75,7,rng) for x in [21,38,55,72]]
        if view==1:return [line((15,61),(77,53),(93,67),(83,84),(18,92),(6,79),(15,61)),line((26,61),(29,34),(64,30),(76,54)),line((58,39),(93,19),(97,25),(65,45)),line((19,68),(81,60),(85,72),(77,79),(20,86),(13,78),(19,68))]
        return [rect(20,40,60,38),rect(25,25,50,30),rect(46,7,8,32),rect(9,39,11,48),rect(80,39,11,48),e(50,44,15,14)]
    if name=='rocket':
        if view==0:return [line((39,65),(39,28),(50,7),(62,28),(62,65),(39,65)),e(50,36,7,8),line((39,49),(22,77),(39,71)),line((62,49),(78,77),(62,71)),line((43,69),(42,90),(50,80),(57,93),(58,69))]
        return [bez((27,72),(22,38),(47,13),(84,12)),bez((84,12),(83,48),(60,75),(27,72)),e(58,39,10,10),line((30,48),(10,63),(27,65)),line((55,69),(48,91),(42,73)),line((24,76),(8,94),(31,82))]
    if name=='nuke':
        if view==0:return [qbez((12,38),(12,6),(50,11)),qbez((50,11),(88,4),(90,39)),qbez((90,39),(49,52),(12,38)),line((39,45),(38,73),(21,88),(80,88),(62,73),(60,45)),e(50,61,24,6)]
        s=[e(50,50,42,42),e(50,50,6,6)]
        for a in [0,TAU/3,2*TAU/3]:s+=[np.concatenate([arc(50,50,14,14,a,a+.85),arc(50,50,35,35,a+.85,a),arc(50,50,14,14,a,a+.05)])]
        return s
    if name=='meteor':
        s=[e(70,69,21,19)]
        if view==0:s += [line((53,56),(9,12)),line((64,48),(39,14)),line((48,68),(10,40))]
        else:s += [line((49,68),(4,16),(38,33),(28,3),(67,48)),e(73,70,5,4),arc(68,67,13,12,.2,1.8)]
        return s
    if name=='volcano':
        s=[line((7,92),(34,46),(63,46),(92,92)),e(49,46,16,4)]
        for x in [33,48,64]:s += [qbez((48,41),(x,16),(x+rng.uniform(-15,15),8))]
        if view:s += [line((39,49),(42,61),(35,72),(44,84)),line((60,49),(55,65),(68,83))]
        return s
    if name=='tsunami':
        if view==0:return [bez((92,89),(100,-10),(5,0),(14,46)),bez((14,46),(18,66),(57,19),(60,50)),qbez((60,50),(59,75),(28,91)),line((7,94),(95,94))]
        return [bez((6,92),(32,77),(31,3),(75,11)),bez((75,11),(101,21),(92,65),(71,47)),bez((71,47),(83,27),(44,25),(47,64)),qbez((47,64),(50,81),(95,91)),line((3,96),(97,96))]
    if name=='dynamite':
        s=[rect(28,37,16,50),rect(45,32,16,55),rect(62,39,15,48),line((25,55),(80,55)),line((25,72),(80,72)),qbez((51,31),(38,8),(71,13))]
        if view:s=[e(31,34,9,5),e(50,30,9,5),e(69,36,9,5)]+s
        return s
    if name=='ninja star':
        n=4 if view==0 else int(rng.integers(3,7));s=[star_poly(50,50,44,12,n,rot=rng.uniform(0,TAU)),e(50,50,5,5)]
        if view:s+=[line((50,50),(50,15)),line((50,50),(79,69))]
        return s
    if name=='bow and arrow':
        s=[qbez((28,8),(94,50),(28,92)),line((28,8),(28,92)),line((8,50),(88,50)),line((74,40),(88,50),(74,60))]
        if view:s[1]=line((28,8),(8,50),(28,92));s+=[line((15,44),(22,50),(15,56))]
        return s
    if name=='shield':
        if view==0:s=[line((19,19),(80,19),(78,64),(50,91),(22,64),(19,19))]
        else:s=[bez((11,24),(20,23),(35,17),(50,7)),bez((50,7),(64,18),(82,23),(90,24)),bez((90,24),(91,71),(65,87),(50,97)),bez((50,97),(23,82),(9,61),(11,24))]
        if view==2:s+=[line((47,28),(55,28),(55,47),(70,47),(70,55),(55,55),(55,76),(47,76),(47,55),(32,55),(32,47),(47,47),(47,28))]
        elif rng.random()<.7:s+=[line((50,24),(50,79)),line((29,43),(72,43))]
        return s
    if name=='ghost':
        s=[bez((20,86),(17,2),(83,3),(80,86)),line((80,86),(65,75),(54,87),(42,77),(30,89),(20,86)),e(38,43,4,7),e(63,43,4,7)]
        if view:s += [e(50,61,7,10),line((20,53),(6,61),(20,65)),line((80,53),(94,60),(80,66))]
        return s
    if name=='robot':
        if view==0:s=[rect(28,15,44,31),rect(26,50,48,33),line((50,15),(50,5)),e(50,4,3,3),rect(35,26,8,8),rect(57,26,8,8),line((28,57),(10,69)),line((74,57),(91,72)),line((36,83),(36,97)),line((63,83),(63,97))]
        else:s=[e(50,30,27,23),rect(31,52,39,30),e(39,27,5,6),e(61,27,5,6),line((35,39),(65,39)),line((30,58),(11,48),(6,59)),line((70,58),(91,48),(96,60)),rect(29,82,12,16),rect(59,82,12,16)]
        return s
    if name in ['alien','zombie','cursed face']:
        if name=='alien':
            s=[bez((50,90),(4,55),(9,6),(50,8)),bez((50,8),(93,5),(96,54),(50,90)),e(33,45,12,7),e(68,45,12,7),line((43,72),(58,72))]
            if view:s+=[line((36,12),(27,0)),line((65,12),(75,0))]
        elif name=='zombie':
            s=[line((22,22),(72,16),(82,76),(61,91),(24,80),(22,22)),e(38,42,8,9),line((62,33),(72,43),(61,49)),line((32,64),(68,64),(66,77),(35,75),(32,64)),line((33,66),(35,74)),line((51,65),(51,76))]
            if view:s+=[line((25,21),(35,5),(49,17)),line((31,53),(23,61)),line((25,57),(29,59))]
        else:s=[e(50,51,39,41),line((20,32),(41,44),(22,47)),line((79,32),(58,44),(78,47)),line((27,67),(37,59),(45,68),(55,59),(64,69),(74,62))]
        return s
    if name=='dinosaur':
        if view==0:return [line((7,66),(28,53),(32,31),(43,17),(72,17),(89,32),(70,43),(57,41),(66,66),(58,83),(45,83),(42,70),(30,80),(20,80),(24,65),(7,66)),e(69,27,3,3),line((61,48),(78,53),(69,58))]
        return [bez((9,73),(24,10),(77,21),(81,59)),line((81,59),(96,42),(92,75),(78,82),(67,81),(66,95),(58,95),(56,80),(36,80),(33,95),(24,95),(24,80),(9,73)),line((17,48),(13,32),(28,38),(29,23),(40,31)),e(21,61,3,3)]
    if name=='poop':
        s=[bez((15,85),(3,69),(20,67),(26,63)),bez((26,63),(15,45),(34,43),(39,39)),bez((39,39),(32,23),(52,25),(54,8)),bez((54,8),(56,33),(74,30),(67,49)),bez((67,49),(89,53),(82,65),(82,65)),bez((82,65),(104,76),(83,91),(15,85))]
        if view:s+=[e(40,63,4,5),e(63,63,4,5),arc(52,66,10,9,0,math.pi)]
        return s
    if name=='heart':
        s=[bez((50,88),(-30,30),(23,-12),(50,30)),bez((50,30),(83,-12),(130,30),(50,88))]
        if view==2:s+=[line((53,23),(44,45),(56,51),(45,75))]
        return s
    if name=='taco':
        s=[arc(50,75,42,55,math.pi,TAU),line((8,75),(92,75)),arc(50,75,34,43,math.pi,TAU)]
        if view:s += [line((20,53),(27,44),(35,49),(43,37),(52,44),(61,36),(70,46),(79,46))]
        return s
    if name=='cheese':
        s=[line((11,67),(71,18),(91,69),(11,67),(11,88),(91,91),(91,69)),line((71,18),(71,40))]
        for cx,cy,r in [(47,54,6),(71,65,5),(30,79,5),(67,81,4)]:s+=[e(cx,cy,r,r*.7)]
        return s
    if name=='straw hat':
        s=[e(50,69,46,13),bez((22,66),(21,5),(76,5),(78,67)),arc(50,58,28,11,0,math.pi)]
        if view:s += [arc(50,69,35,8,0,math.pi),line((31,28),(34,54)),line((45,21),(46,52)),line((60,24),(60,54))]
        return s
    if name=='kunai':
        return [line((50,5),(72,54),(51,66),(28,54),(50,5)),line((50,5),(51,66)),rect(45,66,11,21),e(50,94,8,7)]+([line((45,72),(56,76)),line((45,80),(56,84))] if view else [])
    if name=='spiral':
        t=np.linspace(0,rng.uniform(2.2,4.)*TAU,110);r=np.linspace(1,43,len(t))
        return [np.stack([50+r*np.cos(t),50+r*np.sin(t)*(1 if view==0 else .65)],1)]
    if name=='x2':
        s=[line((9,33),(32,68)),line((32,33),(9,68)),bez((46,38),(57,15),(94,29),(75,46)),line((75,46),(48,70),(87,70))]
        if view==2:s[2]=arc(65,34,19,15,math.pi,TAU+.6)
        return s
    if name=='undo arrow':
        if view==0:return [bez((82,82),(108,23),(55,8),(17,34)),line((17,34),(17,10)),line((17,34),(43,37))]
        return [arc(52,54,35,34,-.25,math.pi*1.55),line((55,19),(34,7)),line((55,19),(37,34))]
    raise KeyError(name)


def make(name,rng,family_id):
    if family_id<0:return THINGS[name](rng)
    strokes=family(name,rng,family_id)
    # Reorder/reverse independent strokes; keep geometry, remove drawing-order signatures.
    if rng.random()<.5:strokes=[s[::-1] if rng.random()<.5 else s for s in strokes]
    if rng.random()<.3:rng.shuffle(strokes)
    return strokes


def job(args):
    ci,n,split,seed=args;name=CATEGORIES[270+ci]
    rng=np.random.default_rng([seed,ci]);x=np.empty((n,4,40,40),np.uint8);regions=np.empty((n,40,40),bool);fills=np.empty(n,np.uint8)
    examples=[]
    from fill import fill_colour
    for i in range(n):
        fid=2 if split=='test' else int(rng.choice([-1,0,1],p=[.1,.45,.45]))
        raw=make(name,rng,fid)
        drawing=to_quickdraw(humanise(raw,rng,name,rng.uniform(.15,1.55)))
        ink,owner,length=render(drawing,40)
        colours=colourise(name,length,rng)
        x[i]=np.rint(planes(ink.astype(np.float32)/255,owner,colours)*255).astype(np.uint8)
        mask=ink>round(.35*255);wall=binary_dilation(mask,structure=np.ones((3,3),bool))
        closed=binary_fill_holes(wall)
        outside=~closed
        outside |= binary_dilation(outside,structure=np.ones((3,3),bool)) & wall & ~mask
        regions[i]=~outside & ~mask
        fills[i]=fill_colour(name,colours,length,rng)
        if i<3:examples.append({'label':270+ci,'family':fid,'drawing':drawing,'colours':[int(c) for c in colours]})
    dest=ROOT/'parts'/f'{split}-{ci:02d}.npz'
    temp=dest.with_suffix('.part')
    with temp.open('wb') as f:np.savez(f,x=x,y=np.full(n,270+ci,np.int16),region=regions,fill=fills)
    os.replace(temp,dest)
    return {'split':split,'category':name,'count':n,'examples':examples}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=int,default=3000)
    ap.add_argument('--validation',type=int,default=200);ap.add_argument('--test',type=int,default=400)
    ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'parts').mkdir(exist_ok=True)
    started=time.time();jobs=[]
    for split,n,seed in [('train',a.train,70161110),('validation',a.validation,80161110),('test',a.test,90161110)]:
        for ci in range(29):jobs.append((ci,n,split,seed))
    records=[]
    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(job,jobs):
            records.append(row);print(row['split'],row['category'],row['count'],round(time.time()-started),flush=True)
    for split in ['train','validation','test']:
        parts=[np.load(ROOT/'parts'/f'{split}-{ci:02d}.npz') for ci in range(29)]
        temp=(ROOT/f'{split}.npz.part')
        with temp.open('wb') as f:np.savez(f,**{k:np.concatenate([z[k] for z in parts]) for k in parts[0].files})
        os.replace(temp,ROOT/f'{split}.npz')
    (ROOT/'manifest.json').write_text(json.dumps({'generator':'training/synth_diverse.py','args':vars(a),
        'source':'Our own handwritten procedural programs only; no outside-set images/strokes.',
        'test':'Seed-disjoint; view=2 held out from train, but some programs share geometry with view=1. Synthetic upper bound.',
        'records':records,'seconds':time.time()-started},indent=1),encoding='utf-8',newline='\n')
    print('DONE',time.time()-started,flush=True)


if __name__=='__main__':main()
