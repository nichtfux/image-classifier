"""Measure actual local generation and end-to-end filter/conversion throughput."""
import argparse,json,time,hashlib,os
from pathlib import Path
import torch,numpy as np
from PIL.PngImagePlugin import PngInfo
from local_flux import Generator,ROOT,REVISION
from local_teacher import Teacher
from generated_prompts import prompt
from generated_convert import convert,preview
from categories import CATEGORIES


def record_quantized():
    root=ROOT/'models/flux1-schnell-nf4';records=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and '.cache' not in p.parts:
            h=hashlib.sha256()
            with p.open('rb') as f:
                while chunk:=f.read(16*1024*1024):h.update(chunk)
            records.append({'file':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    (ROOT/'flux-nf4-derived-hashes.json').write_text(json.dumps({'source_revision':REVISION,'licence':'Apache-2.0 transformer; packaged T5 Apache-2.0','derivation':'bitsandbytes 0.50.2 NF4 double quant, bfloat16 compute','files':records},indent=2),encoding='utf-8',newline='\n')
    p=ROOT/'SOURCES.md';s=p.read_text(encoding='utf-8')
    if '## Locally derived NF4' not in s:
        s+='\n## Locally derived NF4\n\nOriginal files remain unchanged. [Derived weights/configuration hashes](flux-nf4-derived-hashes.json) pin the locally quantised FLUX transformer and packaged T5 encoder; source revision and licences are recorded there. No third-party weights were substituted.\n'
        p.write_text(s,encoding='utf-8',newline='\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group-offload',action='store_true');ap.add_argument('--quick',action='store_true');a=ap.parse_args()
    torch.set_num_threads(5);teacher=Teacher();teacher.to('cpu');torch.cuda.empty_cache()
    gen=Generator(a.group_offload);records=[]
    configs=[(256,1,4),(384,1,4)] if a.quick else [(256,2,4),(384,2,4),(384,2,2),(384,4,2)]
    folder=ROOT/'benchmark';folder.mkdir(exist_ok=True)
    text=prompt('bomb','flat',0);begin=time.time();gen.embeddings(text)
    print('COLD PROMPT ENCODE',time.time()-begin,flush=True)
    for index,(size,batch,steps) in enumerate(configs):
        name='bomb';ci=CATEGORIES.index(name)
        teacher.to('cpu');torch.cuda.empty_cache();start=time.time()
        count=1 if a.quick else 8
        images=[];elapsed=0.
        try:
            for lo in range(0,count,batch):
                part,seconds=gen.generate(text,[71261110+i for i in range(lo,min(lo+batch,count))],size=size,steps=steps)
                images+=part;elapsed+=seconds
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache();print('BENCHMARK OOM',size,batch,steps,flush=True);continue
        generation_wall=time.time()-start
        teacher.to('cuda');rawscores=np.concatenate([teacher.score(images[lo:lo+4]) for lo in range(0,len(images),4)]);kept=0;examples=[]
        for i,image in enumerate(images):
            raw=teacher.decision(rawscores[i],ci);entry={'generated':True,'word':name,'source_teacher':raw}
            info=PngInfo();info.add_text('generated','true');info.add_text('model_revision',REVISION);info.add_text('prompt',text)
            image.save(folder/f'{size}-{batch}-{steps}-{i}-raw.png',pnginfo=info)
            try:
                c=convert(image,'flat',71261110+i);p=preview(c['x']);p.save(folder/f'{size}-{batch}-{steps}-{i}-converted.png',pnginfo=info)
                scores=teacher.score([p])[0];decision=teacher.decision(scores,ci);entry['converted_teacher']=decision
                entry['keep']=raw['keep'] and decision['keep'];kept+=int(entry['keep'])
            except ValueError as e:entry.update({'keep':False,'reason':str(e)})
            examples.append(entry)
        teacher.to('cpu');torch.cuda.empty_cache()
        row={'size':size,'batch':batch,'group_offload':a.group_offload,'steps':steps,'images':len(images),'cached_generation_seconds':elapsed,
             'prompt_and_generation_seconds':generation_wall,'end_to_end_seconds':time.time()-start,
             'generation_images_per_second':len(images)/elapsed,'pipeline_images_per_second':len(images)/(time.time()-start),'kept_per_second':kept/(time.time()-start),'kept':kept,'examples':examples}
        records.append(row);print('BENCHMARK',json.dumps(row),flush=True)
        (ROOT/('benchmark-group.json' if a.group_offload else 'benchmark-resident.json')).write_text(json.dumps(records,indent=2),encoding='utf-8',newline='\n')
    record_quantized();print('BENCHMARK DONE',flush=True)


if __name__=='__main__':main()
