"""Independent scripted regressions, never added to training or used for selection.

Filled objects use dense original mouse-like scan strokes through the public API;
there is no inference-only fill flag. The retained gestures predate this model.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def hatch(polygon,spacing=3):
    result=[]
    for y in range(int(min(p[1] for p in polygon))+2,int(max(p[1] for p in polygon)),spacing):
        crossings=[]
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            if (a[1]<=y<b[1]) or (b[1]<=y<a[1]):
                crossings.append(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
        crossings.sort()
        for x0,x1 in zip(crossings[::2],crossings[1::2]):
            if x1-x0>4:result.append([[round(x0+1,3),y],[round(x1-1,3),y]])
    return result


def cases():
    old={r['name']:r['strokes'] for r in json.loads((ROOT/'notes/mouse-check-299-r3-inputs.json').read_text())['cases']}
    first={r['name']:r['strokes'] for r in json.loads((ROOT/'notes/mouse-check-299-inputs.json').read_text())['cases']}
    result=[]
    def add(name,strokes,colours=None):result.append({'name':name,'strokes':strokes,'colours':colours})
    add('top-fuse bomb coloured',old['bomb'],[1,1,1,5,6,5]);add('top-fuse bomb black',old['bomb'])
    add('apple outline coloured',old['apple'],[4,3,7]);add('apple outline black',old['apple'])
    add('black hole coloured',old['black-hole'],[9,9,1,1,6,6,6,6]);add('spiral black',old['spiral'])
    add('heart outline red',first['heart'],[4])
    banana=[[[42,64],[49,108],[69,148],[104,177],[146,184],[184,171],[210,146],[222,118],[213,134],[183,146],[143,145],[112,129],[89,104],[73,73],[69,52],[42,64]],[[55,59],[53,44],[65,41],[70,53]]]
    add('banana yellow',banana,[6,3]);add('banana blue',banana,[8,3])
    apple=[[115,72],[94,65],[73,72],[61,91],[58,111],[63,142],[78,175],[98,195],[112,197],[126,188],[140,194],[154,185],[171,163],[183,136],[184,109],[174,83],[151,68],[133,68],[115,72]]
    stem=[[122,70],[124,46],[124,41],[132,37]];calyx=[[113,196],[117,193],[121,197],[125,194],[128,198]]
    af=[apple]+hatch(apple)+[stem,calyx]
    add('filled red apple black stem',af,[4]*(len(af)-2)+[1,1])
    add('filled red apple brown stem',af,[4]*(len(af)-2)+[3,1])
    bf=[old['bomb'][0]]+hatch(old['bomb'][0])+old['bomb'][1:]
    add('filled black bomb orange spark',bf,[1]*(len(bf)-3)+[5,6,5])
    db=[old['bomb-diagonal'][0]]+hatch(old['bomb-diagonal'][0])+old['bomb-diagonal'][1:]
    add('filled diagonal bomb orange spark',db,[1]*(len(db)-5)+[5,6,5,6,5])
    hf=first['heart']+hatch(first['heart'][0]);add('filled red heart',hf,[4]*len(hf))
    return result


if __name__=='__main__':
    p=ROOT/'notes/colour-script-inputs.json';p.write_text(json.dumps({'origin':__doc__,'cases':cases()},indent=1),encoding='utf-8',newline='\n')
    print(p)
