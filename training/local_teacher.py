"""Frozen zero-shot SigLIP judge. Outside evaluation sets never tune this model."""
import os,json,time,hashlib
from pathlib import Path
os.environ['HF_HUB_VERBOSITY']='error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SiglipModel,AutoProcessor
from generated_prompts import teacher_texts
from categories import CATEGORIES

ROOT=Path(__file__).resolve().parent.parent/'data/generated'
REVISION='9fdffc58afc957d1a03a25b10dba0329ab15c2a3'


class Teacher:
    def __init__(self,device='cuda'):
        self.device=device;folder=ROOT/'models/siglip-so400m'
        self.processor=AutoProcessor.from_pretrained(folder,local_files_only=True,use_fast=False)
        self.model=SiglipModel.from_pretrained(folder,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
        self.model.to(device)
        texts,negative=teacher_texts()
        signature=hashlib.sha256((REVISION+json.dumps(texts+negative)).encode()).hexdigest()
        cache=ROOT/('teacher-texts-'+signature[:16]+'.pt')
        if cache.exists():self.text=torch.load(cache,map_location=device,weights_only=True)
        else:
            vectors=[]
            with torch.inference_mode():
                for lo in range(0,len(texts)+len(negative),32):
                    batch=(texts+negative)[lo:lo+32]
                    tokens=self.processor(text=batch,padding='max_length',truncation=True,return_tensors='pt').to(device)
                    features=self.model.get_text_features(**tokens)
                    if not torch.is_tensor(features):features=features.pooler_output
                    vectors.append(F.normalize(features.float(),dim=-1))
            vectors=torch.cat(vectors)
            self.text=torch.cat([F.normalize(vectors[:299*4].reshape(299,4,-1).mean(1),dim=-1),vectors[299*4:]])
            torch.save(self.text.cpu(),cache)
        self.text_signature=signature
        self.scale=self.model.logit_scale.exp().detach().float()
        self.bias=self.model.logit_bias.detach().float()
        self.names=CATEGORIES+['blank paper','unrecognisable scribbles','printed text','abstract pattern','multiple-object scene','landscape photograph']
        # Text embeddings are cached; only the vision tower needs GPU memory now.
        self.model.text_model.to('cpu');torch.cuda.empty_cache()

    def to(self,device):
        self.device=device;self.model.vision_model.to(device);self.text=self.text.to(device)
        self.scale=self.scale.to(device);self.bias=self.bias.to(device)
        return self

    @torch.inference_mode()
    def score(self,images):
        pixel=self.processor(images=images,return_tensors='pt')['pixel_values'].to(self.device,dtype=torch.bfloat16)
        features=self.model.get_image_features(pixel_values=pixel)
        if not torch.is_tensor(features):features=features.pooler_output
        cosine=F.normalize(features.float(),dim=-1)@self.text.T
        logits=cosine*self.scale+self.bias
        if not bool(torch.isfinite(logits).all()):raise RuntimeError('Non-finite SigLIP scores; stop before writing labels.')
        return logits.float().cpu().numpy()

    def decision(self,logits,intended,minimum=.05):
        index=int(np.argmax(logits));score=float(1/(1+np.exp(-np.clip(logits[intended],-60,60))))
        other=np.delete(logits,intended).max();margin=float(logits[intended]-other)
        return {'keep':bool(index==intended and score>=minimum),'top_index':index,'top_label':self.names[index],
                'intended_score':score,'margin':margin,'score_minimum':minimum}


if __name__=='__main__':
    import argparse
    from PIL import Image
    ap=argparse.ArgumentParser();ap.add_argument('image');a=ap.parse_args();torch.set_num_threads(3)
    start=time.time();teacher=Teacher();z=teacher.score([Image.open(a.image)])[0]
    ids=np.argsort(-z)[:5];print('TOP',[(teacher.names[int(i)],float(z[i])) for i in ids],'seconds',time.time()-start)
