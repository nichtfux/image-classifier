"""Independent input diagnostics; not a player-accuracy estimate or training data.

Contrast the current normalized thin-stroke view with a conservative brush-aware
view (never narrower than today's lines). No model is trained by this script.
"""
import json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from colour import NET,planes
from colour_raster import reference
from colour_probes import hatch
from train_colour import quant_net

ROOT=Path(__file__).resolve().parent.parent


def ellipse(cx,cy,rx,ry,n=45):
    return [[cx+rx*math.cos(t),cy+ry*math.sin(t)] for t in np.linspace(0,2*math.pi,n)]


def suite():
    # Fixed before prediction; two independently authored styles per concept.
    items=[]
    def add(label,style,strokes,inks,body=None):items.append(dict(label=label,style=style,strokes=strokes,inks=inks,body=body))
    add('apple','round leaf',[ellipse(120,145,62,63),[[120,83],[126,54],[140,43]],[[126,64],[151,46],[169,57],[145,72],[126,64]]],[4,3,7],0)
    add('apple','notched fruit',[[[122,91],[96,79],[73,88],[62,117],[68,151],[89,179],[110,190],[126,181],[145,188],[167,166],[181,133],[176,101],[159,83],[140,80],[122,91]],[[124,87],[121,64],[129,48]]],[7,3],0)
    add('heart','round lobes',[[[125,188],[77,142],[58,111],[66,83],[87,73],[109,80],[125,99],[142,79],[164,73],[184,88],[186,114],[166,145],[125,188]]],[4],0)
    add('heart','slim heart',[[[125,203],[87,143],[71,98],[76,69],[95,63],[115,79],[126,101],[140,78],[157,65],[175,72],[181,95],[163,139],[125,203]]],[10],0)
    add('bomb','cap and fuse',[ellipse(117,151,56,57),[[104,95],[104,83],[128,83],[128,96]],[[116,83],[118,66],[132,61],[137,44]],[[137,44],[134,30]],[[137,44],[153,36]],[[137,44],[152,50]]],[1,1,1,5,6,5],0)
    add('bomb','side fuse',[ellipse(111,149,58,58),[[149,105],[158,91],[173,104],[160,118]],[[165,97],[177,80],[169,66],[186,53]],[[186,53],[184,39]],[[186,53],[202,48]],[[186,53],[199,64]]],[1,1,1,5,6,5],0)
    add('house','square front',[[[66,111],[66,201],[186,201],[186,111]],[[49,116],[125,53],[204,116],[49,116]],[[108,201],[108,151],[141,151],[141,201]],[[79,130],[97,130],[97,149],[79,149],[79,130]]],[3,2,3,8],None)
    add('house','cottage',[[[52,123],[128,72],[198,119]],[[65,114],[65,197],[182,197],[182,111]],[[149,87],[149,57],[169,57],[169,102]],[[106,197],[106,153],[137,153],[137,197]]],[3,3,3,3],None)
    add('tree','round crown',[ellipse(126,105,64,50),[[110,154],[110,213],[139,213],[139,153]]],[7,3],0)
    add('tree','fir',[[[127,42],[80,108],[103,108],[62,156],[106,156],[106,208],[145,208],[145,156],[192,156],[153,108],[176,108],[127,42]]],[7],0)
    add('fish','oval tail',[ellipse(113,132,66,37),[[174,132],[213,96],[213,167],[174,132]],ellipse(75,121,3,3),[[104,102],[128,76],[137,101]]],[8,8,1,8],0)
    add('fish','pointed fish',[[[47,129],[84,92],[135,85],[179,117],[210,94],[207,166],[179,143],[134,173],[82,162],[47,129]],ellipse(75,124,4,4),[[133,91],[143,122],[134,164]]],[5,1,5],0)
    add('star','five points',[[[127,44],[148,103],[211,104],[161,142],[178,204],[127,168],[76,204],[93,142],[42,104],[106,103],[127,44]]],[6],0)
    add('star','four points',[[[127,48],[145,111],[209,129],[145,145],[127,208],[111,145],[48,129],[111,111],[127,48]]],[6],0)
    add('ghost','round sheet',[[[62,205],[64,112],[77,76],[111,59],[146,61],[177,83],[188,113],[188,204],[164,184],[144,205],[123,185],[104,206],[83,184],[62,205]],ellipse(99,118,6,10),ellipse(151,118,6,10),ellipse(125,150,8,11)],[9,1,1,1],0)
    add('ghost','floating blob',[[[67,195],[55,155],[58,98],[83,69],[116,56],[153,65],[178,90],[188,131],[184,174],[168,200],[150,185],[126,205],[106,185],[87,204],[67,195]],[[91,113],[101,113]],[[142,113],[152,113]],[[112,147],[135,149]]],[8,1,1,1],0)
    add('sun','rays',[ellipse(128,128,43,43)]+[[[128+59*math.cos(t),128+59*math.sin(t)],[128+88*math.cos(t),128+88*math.sin(t)]] for t in np.linspace(0,2*math.pi,9)[:-1]],[6]*9,0)
    add('sun','short rays',[ellipse(128,128,66,66)]+[[[128+77*math.cos(t),128+77*math.sin(t)],[128+91*math.cos(t),128+91*math.sin(t)]] for t in np.linspace(0,2*math.pi,13)[:-1]],[5]*13,0)
    add('car','side view',[[[42,145],[69,139],[91,101],[151,101],[183,138],[210,146],[210,179],[43,179],[42,145]],ellipse(82,178,18,18),ellipse(172,178,18,18),[[92,135],[106,110],[142,110],[165,135],[92,135]]],[4,1,1,8],None)
    add('car','boxy hatchback',[[[42,125],[70,125],[88,87],[169,87],[181,125],[207,134],[207,172],[43,172],[42,125]],ellipse(78,173,19,19),ellipse(176,173,19,19),[[95,98],[95,125],[155,125],[155,98],[95,98]]],[8,1,1,8],None)
    add('cat','face',[[[61,163],[54,71],[91,99],[128,90],[163,101],[200,70],[191,166],[164,192],[106,196],[76,182],[61,163]],[[92,135],[102,135]],[[152,135],[162,135]],[[118,151],[130,159],[142,151],[118,151]],[[130,159],[130,175],[117,182]],[[130,175],[145,182]],[[86,159],[43,149]],[[88,170],[41,178]],[[171,159],[213,148]],[[170,172],[212,180]]],[3,1,1,1,1,1,1,1,1,1],0)
    add('cat','body cat',[ellipse(130,91,40,36),[[94,77],[92,43],[119,59]],[[144,59],[173,43],[168,80]],[[102,126],[83,182],[90,210],[118,210],[124,162],[137,210],[169,210],[174,177],[158,125]],[[174,181],[205,163],[205,133],[185,133]],[[113,88],[119,88]],[[144,88],[151,88]],[[124,105],[137,105],[130,111],[124,105]]],[3,3,3,3,3,1,1,1],None)
    add('banana','crescent',[[[52,59],[62,112],[92,158],[131,180],[172,176],[202,151],[218,111],[197,135],[164,145],[126,137],[96,110],[80,75],[76,48],[52,59]]],[6],0)
    add('banana','thin curved',[[[64,47],[73,94],[105,138],[149,157],[192,147],[210,124],[196,170],[165,190],[127,183],[89,153],[65,111],[54,68],[64,47]],[[65,48],[63,35],[72,31],[78,47]]],[6,3],0)
    return items


def capture(strokes,force_end=True):
    out=[]
    for st in strokes:
        kept=[st[0]]
        for p in st[1:-1]:
            if np.linalg.norm(np.asarray(p)-kept[-1])>=3:kept.append(p)
        p=st[-1]
        if force_end or np.linalg.norm(np.asarray(p)-kept[-1])>=3:
            if p!=kept[-1]:kept.append(p)
        out.append(kept)
    return out


def brush_view(raw,widths,size=40,floor_radius=True):
    """Diagnostic only: broaden when visible paint is wider than the model's line."""
    pts=np.concatenate([np.asarray(s,float) for s in raw]);mn,mx=pts.min(0),pts.max(0)
    scale=(22*size/28)/max(mx-mn);centre=(mn+mx)/2
    yy,xx=np.mgrid[:size,:size];px,py=xx+.5,yy+.5
    ink=np.zeros((size,size));owner=np.zeros((size,size),np.int64)
    for si,st in enumerate(raw):
        st=(np.asarray(st)-centre)*scale+size/2;best=np.full((size,size),np.inf)
        for a,b in zip(st,st[1:] if len(st)>1 else st):
            dx,dy=b-a;ll=dx*dx+dy*dy;t=np.clip(((px-a[0])*dx+(py-a[1])*dy)/ll,0,1) if ll>1e-12 else 0
            best=np.minimum(best,(px-a[0]-t*dx)**2+(py-a[1]-t*dy)**2)
        alpha=np.clip(max(size/28 if floor_radius else 0,widths[si]*scale/2)+.5-np.sqrt(best),0,1)
        wins=(alpha>0)&(alpha>=ink);owner[wins]=si;ink=np.maximum(ink,alpha)
    return ink,owner


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=ROOT/'training/out';meta=json.loads((out/'colour299_40.json').read_text());labels=meta['categories']
    with np.load(out/'colour299_40.npz') as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    net=quant_net(q,'cuda');records=[];images=[];gallery=[]
    for base in suite():
        for extent in (80,160,320):
            for paint in (False,True):
                if paint and base['body'] is None:continue
                for coloured in (False,True):
                    raw=[np.asarray(s,float).tolist() for s in base['strokes']];inks=list(base['inks'])
                    if paint:
                        # Broad quick back-and-forth fill, spaced in screen pixels.
                        body=raw[base['body']];lines=hatch(body,spacing=max(1,round(12*256/extent)))
                        raw=raw[:1]+lines+raw[1:];inks=inks[:1]+[inks[base['body']]]*len(lines)+inks[1:]
                    raw=[((np.asarray(s)-128)*extent/256+220).tolist() for s in raw]
                    raw=capture(raw);colours=[c-1 if coloured else 0 for c in inks]
                    drawing=[[[p[0] for p in s],[p[1] for p in s]] for s in raw]
                    ink,owner=reference(drawing,40);wide,wo=brush_view(raw,[16 if paint else 4]*len(raw))
                    for view,i,o in [('current',ink,owner),('brush_floor_probe',wide,wo)]:
                        records.append({'label':base['label'],'style':base['style'],'extent':extent,'paint':paint,'coloured':coloured,'view':view})
                        images.append(planes(i,o,colours).astype(np.float32))
                    if base['label'] in ('apple','heart','bomb') and paint and coloured and extent==80:
                        gallery.append((base['label']+' / '+base['style'],planes(ink,owner,colours),planes(wide,wo,colours)))
    x=torch.from_numpy(np.stack(images)).to('cuda').contiguous(memory_format=torch.channels_last)
    with torch.no_grad():logits=net(x).cpu().numpy()
    for r,z in zip(records,logits):
        ids=np.argsort(-z)[:3];exp=np.exp(z-z.max());p=exp/exp.sum()
        r['top']=[{'label':labels[int(i)],'prob':float(p[i])} for i in ids]
        r['correct']=r['top'][0]['label']==r['label'];r['top3']=any(t['label']==r['label'] for t in r['top'])
        r['custom_alarm']=bool(labels.index(r['label'])<270 and ids[0]>=270)
        shape=z.copy();shape[270:]=-np.inf;r['without_custom']=labels[int(shape.argmax())]
    groups={}
    for view in ('current','brush_floor_probe'):
        for paint in (False,True):
            for extent in (80,160,320):
                s=[r for r in records if r['view']==view and r['paint']==paint and r['extent']==extent]
                groups[f'{view}/'+('paint' if paint else 'outline')+f'/{extent}']={'n':len(s),'top1':float(np.mean([r['correct'] for r in s])),'top3':float(np.mean([r['top3'] for r in s])),'custom_alarms':sum(r['custom_alarm'] for r in s)}
    report={'caveat':'Self-authored diagnostic proxies, not representative player accuracy. Fixed before scoring; never used for training. Brush-aware rendering is an uninstalled probe, not a validated fix.','groups':groups,'cases':records}
    (ROOT/'notes/live-input-audit.json').write_text(json.dumps(report,indent=1),encoding='utf-8',newline='\n')
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',n)
    im=Image.new('RGB',(650,75+len(gallery)*180),'white');d=ImageDraw.Draw(im)
    d.text((15,8),'Small broad-brush paint: two network views',font=font(22),fill='black')
    d.text((15,40),'Current fixed line width             Brush-aware diagnostic (not installed)',font=font(15),fill='#444444')
    for row,(name,a,b) in enumerate(gallery):
        y=75+row*180;d.text((15,y),name,font=font(17),fill='black')
        for col,x in enumerate((a,b)):
            rgb=x[1:].transpose(1,2,0)+(1-x[0,...,None])*np.array([246,241,230])/255
            tile=Image.fromarray(np.rint(rgb*255).clip(0,255).astype(np.uint8)).resize((140,140),Image.Resampling.NEAREST)
            im.paste(tile,(60+col*330,y+25))
    im.save(ROOT/'notes/live-input-brush-probe.png');print(json.dumps(groups,indent=1))


if __name__=='__main__':main()
