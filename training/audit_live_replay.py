"""Replay captured MCP strokes; compare Lua/Python pixels and painted interiors.

Only reconstruct enclosed paper pixels that the actual displayed brush painted
opaquely. Keep normalized thin outlines/details; do not globally thicken them.
No labels or colour priors are consulted by this preprocessing.
"""
import json
from pathlib import Path
import numpy as np
import torch
from colour import planes
from colour_raster import reference
from fill import inside
from audit_live_input import brush_view
from train_colour import quant_net

ROOT=Path(__file__).resolve().parent.parent


def reconstruct(raw,colours,widths,size=40):
    drawing=[[[p[0] for p in s],[p[1] for p in s]] for s in raw]
    ink,owner=reference(drawing,size);original=planes(ink,owner,colours)
    paint,po=brush_view(raw,widths,size,floor_radius=False)
    mask=inside(torch.from_numpy(ink)[None]).numpy()[0] & (paint>=.9)
    corrected=original.copy();corrected[0,mask]=1
    rgb=planes(np.ones_like(ink),po,colours)
    corrected[1:,mask]=rgb[1:,mask]
    return original,corrected,int(mask.sum())


if __name__=='__main__':
    torch.set_num_threads(4);out=ROOT/'training/out'
    meta=json.loads((out/'colour299_40.json').read_text());labels=meta['categories']
    with np.load(out/'colour299_40.npz') as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    net=quant_net(q,'cuda');records=[];images=[];worst=0
    for p in sorted((ROOT/'notes').glob('live-paced-before-*.json')):
        row=json.loads(p.read_text());cap=row['capture']['strokes'];raw=[s['points'] for s in cap]
        cols=[s['colour']-1 for s in cap];widths=[s['width'] for s in cap]
        original,fixed,n=reconstruct(raw,cols,widths)
        # Original Lua planes are padded; compare every channel/pixel, not sums.
        lua=np.asarray(row['image']).reshape(4,42,42)[:,1:-1,1:-1]
        delta=float(abs(lua-original).max());worst=max(worst,delta)
        images.extend([original,fixed]);records.append({'file':p.name,'label':row['case']['label'],'style':row['case']['style'],'size':row['case']['size'],'paint':row['case']['paint'],'changed_pixels':n,'max_lua_python_pixel_diff':delta,'original_luau_top':row['top']})
    with torch.no_grad():p=net(torch.from_numpy(np.stack(images)).float().cuda().contiguous(memory_format=torch.channels_last)).softmax(1).cpu().numpy()
    for i,r in enumerate(records):
        for mode,values in [('original',p[2*i]),('painted_interiors',p[2*i+1])]:
            ids=np.argsort(-values)[:3];r[mode]=[{'label':labels[int(j)],'prob':float(values[j])} for j in ids]
    result={'caveat':'Twelve self-authored MCP development cases, not a representative player test. No training or class priors used. Diagnostic preprocessing was tested and not installed.','max_lua_python_pixel_diff':worst,'records':records}
    (ROOT/'notes/live-replay-audit.json').write_text(json.dumps(result,indent=1),encoding='utf-8',newline='\n')
    for r in records:print(r['label'],r['size'],'paint',r['paint'],r['original'][0]['label'],'->',r['painted_interiors'][0]['label'],'pixels',r['changed_pixels'])
    print('max Lua/Python pixel difference',worst)
