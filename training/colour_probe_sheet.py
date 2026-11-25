"""Numerical-input comparison figure, using only independent development strokes."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from colour import NET
from colour_probes import cases,ROOT
from colour_raster import reference


def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',size)


if __name__=='__main__':
    names=['filled red apple black stem','filled diagonal bomb orange spark','filled red heart','filled black bomb orange spark']
    models=[('r4 / 28px',28,False,'studio-colour-probes-r4.json'),('Colour + fill / 28px',28,True,'studio-colour-probes-28.json'),('Colour + fill / 40px',40,True,'studio-colour-probes-40.json')]
    probes={c['name']:c for c in cases()};reports=[]
    for _,_,_,p in models:
        reports.append({c['name']:c for c in json.loads((ROOT/'notes'/p).read_text())['cases']})
    image=Image.new('RGB',(1020,1660),'white');draw=ImageDraw.Draw(image)
    draw.text((28,15),'Filled-shape regressions: identical source strokes',font=font(25,True),fill='#161a23')
    draw.text((28,51),'Network inputs at the same display size; first / second / third guesses below.',font=font(18),fill='#555b64')
    for j,(label,*_) in enumerate(models):draw.text((28+j*330,85),label,font=font(22,True),fill='#161a23')
    for row,name in enumerate(names):
        y=125+row*375;c=probes[name]
        draw.text((28,y),name,font=font(20,True),fill='#252a34')
        raw=[[[p[0] for p in st],[p[1] for p in st]] for st in c['strokes']]
        for j,(_,size,coloured,_) in enumerate(models):
            ink,owner=reference(raw,size);colours=[v-1 for v in c['colours']]
            rgb=NET[np.asarray(colours)[owner]] if coloured else np.zeros((size,size,3))
            paper=np.array([245,241,228])/255
            pixels=(ink[...,None]*rgb+(1-ink[...,None])*paper)*255
            tile=Image.fromarray(np.rint(pixels).clip(0,255).astype(np.uint8)).resize((280,280),Image.Resampling.NEAREST)
            image.paste(tile,(28+j*330,y+32))
            top=reports[j][name]['top']
            draw.text((28+j*330,y+315),f"{top[0]['label']} {top[0]['prob']:.0%}",font=font(21,True),fill='#161a23')
            draw.text((28+j*330,y+343),f"{top[1]['label']} {top[1]['prob']:.0%} / {top[2]['label']} {top[2]['prob']:.0%}",font=font(16),fill='#555b64')
    draw.text((28,1631),'Development regressions, never trained on. These scores are not a human accuracy estimate.',font=font(17),fill='#555b64')
    path=ROOT/'notes/colour-probes-side-by-side.png';image.save(path);print(path)
