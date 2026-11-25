"""Resumable, labelled FLUX/SigLIP dataset. Local development/training only.

No network calls during inference. Original and converted teacher top word must
match the intended label; all decisions, including rejects, remain in a manifest.
"""
import argparse,json,time,os,hashlib,gc
from pathlib import Path
import numpy as np
import torch
from PIL.PngImagePlugin import PngInfo
from categories import CATEGORIES
from generated_prompts import STYLES,prompt
from generated_convert import convert,preview,VERSION
from local_teacher import Teacher,REVISION as TEACHER_REV
from local_flux import Generator,REVISION as FLUX_REV,ROOT
from generated_splits import assign,group_key,METHOD


def read_rows():
    path=ROOT/'manifest.jsonl'
    if not path.exists():return []
    rows=[]
    for line in path.open(encoding='utf-8'):
        if line.strip():rows.append(json.loads(line))
    return rows


def counts(rows):
    attempts=np.zeros(299,np.int64);kept=np.zeros(299,np.int64);styles={}
    for row in rows:
        ci=row['label'];attempts[ci]+=1;kept[ci]+=int(row['keep'])
        key=(ci,row['style']);a,k=styles.get(key,(0,0));styles[key]=(a+1,k+int(row['keep']))
    return attempts,kept,styles


def state(rows,started,initial,seconds,statistics=None,initial_kept=0):
    a,k,styles=counts(rows) if statistics is None else statistics;total=len(rows)-initial
    keep=int(k.sum())-initial_kept
    result={'generated':True,'model':'black-forest-labs/FLUX.1-schnell','revision':FLUX_REV,
       'teacher':'google/siglip-so400m-patch14-384','teacher_revision':TEACHER_REV,
       'attempted':len(rows),'kept':int(k.sum()),'session_attempted':total,'session_kept':keep,
       'session_seconds':seconds,'images_per_second':total/max(seconds,1e-9),'kept_per_second':keep/max(seconds,1e-9),
       'classes':[{'label':name,'attempted':int(a[i]),'kept':int(k[i]),'keep_rate':float(k[i]/a[i]) if a[i] else None,
           'styles':{s:{'attempted':styles.get((i,s),(0,0))[0],'kept':styles.get((i,s),(0,0))[1]} for s in STYLES}}
           for i,name in enumerate(CATEGORIES)],'status':'running','pid':os.getpid()}
    path=ROOT/'generation-status.json';temp=path.with_suffix('.part')
    temp.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    # Windows readers and scanners can briefly deny replacement. A status-file
    # collision must not abort generation after the manifest has been flushed.
    for retry in range(10):
        try:
            os.replace(temp,path)
            break
        except PermissionError:
            time.sleep(.1)
    else:
        print('Status update deferred after Windows file lock; manifest is current.',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group-offload',action='store_true')
    ap.add_argument('--size',type=int,default=384);ap.add_argument('--batch',type=int,default=2)
    ap.add_argument('--steps',type=int,default=4);ap.add_argument('--stop-after-attempts',type=int,default=0)
    ap.add_argument('--stop-after-kept',type=int,default=0);ap.add_argument('--custom-target',type=int,default=5000)
    ap.add_argument('--real-target',type=int,default=100);ap.add_argument('--max-attempts-per-class',type=int,default=30000)
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    # OS-held lock survives neither exit nor crash; a second worker cannot race IDs.
    lock=(ROOT/'generation.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as e:raise RuntimeError('Another generated-data worker already owns this dataset.') from e
    rows=read_rows();initial=len(rows);initial_kept=sum(r['keep'] for r in rows)
    attempted,kept,style_counts=counts(rows)
    seen_inputs=set()
    for row in rows:
        if row['keep']:
            signature=row.get('input_sha256')
            if signature is None:
                with np.load(ROOT/row['sample']) as z:signature=hashlib.sha256(z['x'].tobytes()).hexdigest()
            seen_inputs.add(signature)
    sources={n:Path(__file__).with_name(n).read_bytes() for n in ['generated_prompts.py','generated_convert.py','generated_splits.py','local_teacher.py','local_flux.py','generate_dataset.py','colour.py','colour_raster.py','fast_raster.py','categories.py']}
    signature=hashlib.sha256(b''.join(n.encode()+sources[n] for n in sorted(sources))).hexdigest()
    snapshot=ROOT/'pipeline_snapshots'/signature;snapshot.mkdir(parents=True,exist_ok=True)
    for n,content in sources.items():(snapshot/n).write_bytes(content)
    teacher=Teacher();teacher.to('cpu');torch.cuda.empty_cache()
    gen=Generator(a.group_offload)
    for name in ['raw','converted','samples']:(ROOT/name).mkdir(exist_ok=True)
    started=time.time();last=time.time();round_index=0
    manifest=(ROOT/'manifest.jsonl').open('a',encoding='utf-8',newline='\n')
    finished=False
    while not finished:
        # Five custom styles, then a stratified real block, including in the pilot.
        real_cycle=round_index%6==5
        block=(round_index//6)%5
        indices=list(range(block,270,5)) if real_cycle else list(range(270,299))
        if real_cycle and block==0:
            priority=[CATEGORIES.index(n) for n in ['apple','banana','cat','bear','car','face'] if n in CATEGORIES]
            indices=list(dict.fromkeys(priority+indices))
        any_job=False
        for ci in indices:
            target=a.real_target if ci<270 else a.custom_target
            if kept[ci]>=target or attempted[ci]>=a.max_attempts_per_class:continue
            if attempted[ci]>=100 and kept[ci]==0:continue # flag an unusable judge/prompt, never spin forever
            name=CATEGORIES[ci];available=(['flat','painted'] if ci%2==0 else ['painted','flat']) if ci<270 else list(STYLES)
            # Balance attempts across the requested styles, rather than letting easy styles dominate.
            style=min(available,key=lambda s:style_counts.get((ci,s),(0,0))[0])
            index=int(attempted[ci]);text=prompt(name,style,index//a.batch);n=a.batch
            seeds=[202611100000+ci*1000000+index+j for j in range(n)]
            any_job=True;teacher.to('cpu');torch.cuda.empty_cache();begin=time.time()
            images,network_seconds=gen.generate(text,seeds,size=a.size,steps=a.steps)
            torch.cuda.empty_cache();teacher.to('cuda')
            originals=teacher.score(images)
            converted_parts=[];converted_images=[];converted_indices=[]
            for j,(image,seed) in enumerate(zip(images,seeds)):
                try:
                    part=convert(image,style,seed+77);converted=preview(part['x'])
                    converted_parts.append((part,converted,None));converted_images.append(converted);converted_indices.append(j)
                except ValueError as error:converted_parts.append((None,None,str(error)))
            converted_scores={}
            for start in range(0,len(converted_images),4):
                z=teacher.score(converted_images[start:start+4])
                converted_scores.update(zip(converted_indices[start:start+4],z))
            for j,(image,seed) in enumerate(zip(images,seeds)):
                identity=f'generated-{ci:03d}-{index+j:07d}'
                rawpath=ROOT/'raw'/f'{ci:03d}';rawpath.mkdir(exist_ok=True)
                rawpath=rawpath/(identity+'.png')
                info=PngInfo();info.add_text('generated','true');info.add_text('model','FLUX.1-schnell');info.add_text('revision',FLUX_REV);info.add_text('seed',str(seed));info.add_text('prompt',text)
                image.save(rawpath,pnginfo=info)
                row={'id':identity,'generated':True,'model':'FLUX.1-schnell','model_revision':FLUX_REV,
                     'teacher_revision':TEACHER_REV,'label':ci,'word':name,'style':style,'prompt':text,'seed':seed,
                     'width':a.size,'height':a.size,'steps':a.steps,'quantization':'local NF4 double quant, bfloat16 compute',
                     'raw_png':rawpath.relative_to(ROOT).as_posix(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     'source_teacher':teacher.decision(originals[j],ci),'keep':False,'conversion_version':VERSION}
                row['pipeline_snapshot']=signature;row['teacher_text_signature']=teacher.text_signature
                row['batch_size']=a.batch;row['prompt_index']=index//a.batch
                row['raw_sha256']=hashlib.sha256(rawpath.read_bytes()).hexdigest()
                try:
                    c,converted,error=converted_parts[j]
                    if error:raise ValueError(error)
                    z=converted_scores[j]
                    row['converted_teacher']=teacher.decision(z,ci)
                    row['stroke_count']=c['stroke_count'];row['point_count']=c['point_count']
                    row['work_length_to_span']=c['work_length_to_span']
                    row['input_sha256']=hashlib.sha256(c['x'].tobytes()).hexdigest()
                    row['keep']=row['source_teacher']['keep'] and row['converted_teacher']['keep']
                    row['reason']='kept' if row['keep'] else ('source_teacher' if not row['source_teacher']['keep'] else 'converted_teacher')
                    if row['keep'] and row['input_sha256'] in seen_inputs:row['keep']=False;row['reason']='duplicate_converted_input'
                    if row['keep']:seen_inputs.add(row['input_sha256'])
                    imgpath=ROOT/'converted'/(identity+'.png');converted.save(imgpath,pnginfo=info);row['converted_png']=imgpath.relative_to(ROOT).as_posix()
                    if row['keep']:
                        row['split']=assign(row);row['split_group']=group_key(row);row['split_method']=METHOD
                        sample=ROOT/'samples'/(identity+'.npz')
                        np.savez_compressed(sample,x=c['x'],y=np.int16(ci),teacher_logits=z[:299].astype(np.float16),generated=np.bool_(True),source_id=np.asarray(identity))
                        row['sample']=sample.relative_to(ROOT).as_posix()
                        row['sample_sha256']=hashlib.sha256(sample.read_bytes()).hexdigest()
                        geometry=ROOT/'samples'/(identity+'.json')
                        geometry.write_text(json.dumps({'generated':True,'format':'QuickDraw [xs,ys], object coordinates 0..256; DrawPalette indices are 1-based','strokes':c['strokes'],'colours':[v+1 for v in c['colours']],'palette_base':1,'display_widths_in_object_coordinates':c['widths'],'conversion_version':VERSION},separators=(',',':')),encoding='utf-8',newline='\n')
                except ValueError as error:row['reason']='conversion: '+str(error)
                rows.append(row);manifest.write(json.dumps(row,separators=(',',':'))+'\n');manifest.flush()
                attempted[ci]+=1;kept[ci]+=int(row['keep'])
                key=(ci,style);ca,ck=style_counts.get(key,(0,0));style_counts[key]=(ca+1,ck+int(row['keep']))
            teacher.to('cpu');torch.cuda.empty_cache()
            status=state(rows,started,initial,time.time()-started,(attempted,kept,style_counts),initial_kept)
            if time.time()-last>45:
                print('GENERATION',status['attempted'],'attempted',status['kept'],'kept',round(status['images_per_second'],3),'images/s',flush=True);last=time.time()
            if ((a.stop_after_attempts and len(rows)-initial>=a.stop_after_attempts) or
                (a.stop_after_kept and status['kept']-initial_kept>=a.stop_after_kept) or (ROOT/'STOP_GENERATION').exists()):
                finished=True;break
        round_index+=1
        if not any_job:
            targets=np.r_[np.full(270,a.real_target),np.full(29,a.custom_target)]
            eligible=(kept<targets)&(attempted<a.max_attempts_per_class)&~((attempted>=100)&(kept==0))
            if not eligible.any():break
    status=state(rows,started,initial,time.time()-started,(attempted,kept,style_counts),initial_kept);status['status']='stopped_cleanly'
    (ROOT/'generation-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8',newline='\n')
    print('GENERATION STOPPED CLEANLY',status['session_attempted'],status['session_kept'],status['session_seconds'],flush=True)


if __name__=='__main__':main()
