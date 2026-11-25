"""Own generated validation only: can smooth teacher presentation rescue detail?

Does not change student pixels, manifests, acceptance, or any external test.
"""
import json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from generated_splits import assign
from local_teacher import Teacher,ROOT


if __name__=='__main__':
    torch.set_num_threads(5)
    rows=[];seen=set()
    for line in (ROOT/'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);key=(r['word'],r['style'])
        if assign(r)=='validation' and r.get('converted_png') and r['source_teacher']['keep'] and key not in seen:
            rows.append(r);seen.add(key)
        if len(rows)>=40:break
    teacher=Teacher(device='cpu');teacher.model.vision_model.float()
    results=[];start=time.time()
    with torch.inference_mode():
        for r in rows:
            im=Image.open(ROOT/r['converted_png']).convert('RGB').resize((40,40),Image.Resampling.NEAREST)
            images=[im.resize((384,384),mode) for mode in [Image.Resampling.NEAREST,Image.Resampling.BICUBIC]]
            pixel=teacher.processor(images=images,return_tensors='pt')['pixel_values'].float()
            h=teacher.model.get_image_features(pixel_values=pixel)
            if not torch.is_tensor(h):h=h.pooler_output
            z=(F.normalize(h.float(),dim=-1)@teacher.text.T*teacher.scale+teacher.bias).numpy()
            out={'id':r['id'],'word':r['word'],'style':r['style'],'split':'validation','nearest':teacher.decision(z[0],r['label']),'bicubic':teacher.decision(z[1],r['label'])}
            results.append(out);print(r['word'],r['style'],out['nearest']['top_label'],out['bicubic']['top_label'],'seconds',round(time.time()-start),flush=True)
    report={'scope':'Own generated prompt-disjoint validation; source teacher passed. Float32 CPU vision. No filter changed.',
            'n':len(rows),'seconds':time.time()-start,'nearest_kept':sum(r['nearest']['keep'] for r in results),
            'bicubic_kept':sum(r['bicubic']['keep'] for r in results),'rows':results}
    (ROOT/'teacher-resampling-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    print('DONE',report['n'],report['nearest_kept'],report['bicubic_kept'],report['seconds'],flush=True)
