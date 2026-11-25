"""Pinned local-only downloads. HF auth uses the hub's normal token lookup only.

Never read/print/copy the token file or put credentials in model metadata/logs.
"""
import hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata

os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_HUB_VERBOSITY']='error'
from huggingface_hub import snapshot_download,HfApi

ROOT=Path(__file__).resolve().parent.parent/'data/generated'
MODELS=[
    ('flux1-schnell','black-forest-labs/FLUX.1-schnell','741f7c3ce8b383c54771c7003378a50191e9efe9',
     ['README.md','model_index.json','scheduler/*','text_encoder/*','text_encoder_2/*','tokenizer/*','tokenizer_2/*','transformer/*','vae/*']),
    ('siglip-so400m','google/siglip-so400m-patch14-384','9fdffc58afc957d1a03a25b10dba0329ab15c2a3',['*.json','*.model','*.safetensors','README.md'])]


def download(row):
    name,repo,revision,patterns=row;folder=ROOT/'models'/name
    # True requests library-managed default token lookup; its value is never accessed here.
    info=HfApi().model_info(repo,revision=revision,token=True)
    print('DOWNLOAD',name,'revision',info.sha,flush=True)
    snapshot_download(repo,revision=revision,local_dir=folder,allow_patterns=patterns,token=True,max_workers=3)
    records=[]
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '.cache' in path.parts:continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            while chunk:=f.read(16*1024*1024):h.update(chunk)
        relative=path.relative_to(folder).as_posix()
        records.append({'file':relative,'bytes':path.stat().st_size,'sha256':h.hexdigest(),
                        'url':f'https://huggingface.co/{repo}/resolve/{revision}/{relative}'})
    result={'name':name,'repo':repo,'revision':revision,'licence':info.card_data.get('license') if info.card_data else None,
            'files':records,'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (ROOT/(name+'-download.json')).write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print('DOWNLOADED',name,sum(r['bytes'] for r in records),'bytes',len(records),'files',flush=True)
    return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(download,MODELS))
    versions={p:importlib.metadata.version(p) for p in ['torch','diffusers','transformers','huggingface_hub','accelerate','bitsandbytes','safetensors','sentencepiece']}
    (ROOT/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8',newline='\n')
    lines=['# Local model sources and generated-data provenance','',
        'The author approved local SigLIP and FLUX.1-schnell on 2026-11-10 and personally accepted the FLUX access conditions. Downloads use huggingface_hub normal token lookup; credentials are never copied into the project. No paid service is used.','',
        'All images produced by FLUX are generated training/development data, never player-facing assets. Every sample records generated=true, model/revision, prompt, seed, style, intended class, teacher scores/filter result and conversion version.','',
        'SigLIP is a frozen zero-shot teacher, not fine-tuned on any outside test set. Its source model card reports WebLI pretraining. TU Berlin, DomainNet, QD recognised=false and the author held-outs remain test-only.','']
    for result in results:
        lines += [f"## {result['repo']}",'',f"Revision: `{result['revision']}`. Licence: `{result['licence']}`. [Official model card](https://huggingface.co/{result['repo']}).",'',
                  '| File | Bytes | SHA-256 | Download |','|---|---:|---|---|']
        for r in result['files']:lines.append(f"| {r['file']} | {r['bytes']} | `{r['sha256']}` | [pinned URL]({r['url']}) |")
        lines.append('')
    lines += ['## FLUX pipeline components','',
        'The FLUX transformer and VAE use the BFL FLUX.1-schnell repository revision above (Apache-2.0). The packaged text encoders are CLIP-L (upstream OpenAI CLIP, MIT) and T5-v1.1-XXL (upstream Google T5, Apache-2.0); the exact downloaded packaged files are pinned and hashed above. These encoders are not additional generative services. Four-bit NF4 is applied locally for the 12 GB GPU; original downloaded safetensors remain unchanged.','',
        '- [OpenAI CLIP source/licence](https://github.com/openai/CLIP/blob/main/LICENSE)','- [Google T5 source/licence](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/LICENSE)','',
        '## Software versions','',*['- '+k+': '+v for k,v in versions.items()],'']
    (ROOT/'SOURCES.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('ALL DOWNLOADS AND HASHES COMPLETE',flush=True)


if __name__=='__main__':main()
