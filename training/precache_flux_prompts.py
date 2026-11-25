"""Precompute local text embeddings while the control owns most GPU memory.

Loads only T5/CLIP, never the large image transformer. No images are generated.
"""
import os,time,hashlib,json
from pathlib import Path
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_HUB_VERBOSITY']='error'
import torch
from diffusers import FluxPipeline
from transformers import T5EncoderModel
from generated_prompts import prompt,STYLES
from categories import CATEGORIES

ROOT=Path(__file__).resolve().parent.parent/'data/generated';REV='741f7c3ce8b383c54771c7003378a50191e9efe9'

if __name__=='__main__':
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    torch.set_num_threads(3)
    text=T5EncoderModel.from_pretrained(ROOT/'models/flux1-schnell-nf4/text_encoder_2',local_files_only=True,torch_dtype=torch.bfloat16,device_map='cpu')
    pipe=FluxPipeline.from_pretrained(ROOT/'models/flux1-schnell',transformer=None,vae=None,text_encoder_2=text,local_files_only=True,torch_dtype=torch.bfloat16)
    pipe.text_encoder.to('cuda');pipe.text_encoder_2.to('cuda')
    texts=[]
    for name in CATEGORIES[270:]:
        for attempt in [0,5,10,15,20,25,30,35,40,45]:
            for style in STYLES:texts.append(prompt(name,style,attempt))
    for name in CATEGORIES[:270]:
        for style in ['flat','painted']:texts.append(prompt(name,style,0))
    folder=ROOT/'prompt_cache';folder.mkdir(exist_ok=True)
    unique=list(dict.fromkeys(texts));jobs=[]
    for text in unique:
        key=hashlib.sha256((REV+'|128|'+text).encode()).hexdigest();path=folder/(key+'.pt')
        if not path.exists():jobs.append((text,path))
    start=time.time();last=start;done=0
    with torch.inference_mode():
        for lo in range(0,len(jobs),2):
            batch=jobs[lo:lo+2]
            embeddings,pooled,_=pipe.encode_prompt(prompt=[s for s,_ in batch],device=torch.device('cuda'),num_images_per_prompt=1,max_sequence_length=128)
            embeddings=embeddings.cpu();pooled=pooled.cpu()
            for i,(_,path) in enumerate(batch):
                if not path.exists():
                    temp=path.with_suffix('.part')
                    torch.save({'embeddings':embeddings[i:i+1].clone(),'pooled':pooled[i:i+1].clone()},temp);os.replace(temp,path)
                done+=1
            if time.time()-last>45:
                print('PROMPT CACHE',done,'/',len(jobs),'seconds',time.time()-start,flush=True);last=time.time()
    print('PROMPT CACHE DONE',done,time.time()-start,flush=True)
