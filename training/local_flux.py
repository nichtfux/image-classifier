"""Offline FLUX.1-schnell, locally NF4 quantised for the 12 GB GPU."""
import os,json,hashlib,time,gc
from pathlib import Path
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_HUB_VERBOSITY']='error'
if os.name=='nt':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
import torch
from diffusers import FluxPipeline,FluxTransformer2DModel,BitsAndBytesConfig as DConfig
from transformers import T5EncoderModel,BitsAndBytesConfig as TConfig

ROOT=Path(__file__).resolve().parent.parent/'data/generated'
REVISION='741f7c3ce8b383c54771c7003378a50191e9efe9'


class Generator:
    def __init__(self,group_offload=False):
        if group_offload:
            raise RuntimeError('Quantised group offload is disabled on this driver stack; wait for the control and use resident NF4.')
        root=ROOT/'models/flux1-schnell';quant=ROOT/'models/flux1-schnell-nf4'
        ready=(quant/'ready.json').exists();start=time.time()
        if ready:
            if json.loads((quant/'ready.json').read_text())['source_revision']!=REVISION:
                raise RuntimeError('Local NF4 cache belongs to a different FLUX revision; do not silently reuse it.')
            transformer=FluxTransformer2DModel.from_pretrained(quant/'transformer',local_files_only=True,torch_dtype=torch.bfloat16,device_map='cpu')
            text=T5EncoderModel.from_pretrained(quant/'text_encoder_2',local_files_only=True,torch_dtype=torch.bfloat16,device_map='cpu')
        else:
            settings=dict(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
            transformer=FluxTransformer2DModel.from_pretrained(root,subfolder='transformer',local_files_only=True,
                quantization_config=DConfig(**settings),torch_dtype=torch.bfloat16,device_map='cpu')
            print('FLUX transformer quantised on CPU',time.time()-start,flush=True)
            text=T5EncoderModel.from_pretrained(root,subfolder='text_encoder_2',local_files_only=True,
                quantization_config=TConfig(**settings),torch_dtype=torch.bfloat16,device_map='cpu')
            quant.mkdir(parents=True,exist_ok=True)
            transformer.save_pretrained(quant/'transformer');text.save_pretrained(quant/'text_encoder_2')
            (quant/'ready.json').write_text(json.dumps({'source_revision':REVISION,'quantization':settings,'created':time.time()},default=str),encoding='utf-8',newline='\n')
        self.pipe=FluxPipeline.from_pretrained(root,transformer=transformer,text_encoder_2=text,
            local_files_only=True,torch_dtype=torch.bfloat16)
        self.pipe.set_progress_bar_config(disable=True);self.pipe.vae.enable_slicing()
        self.group=group_offload;self.pipe.vae.to('cuda')
        if group_offload:
            transformer.enable_group_offload(onload_device=torch.device('cuda'),offload_type='block_level',num_blocks_per_group=1,use_stream=False)
            # Generic group hooks move parameter data, not bitsandbytes' nested
            # quantisation metadata. Keep those small scale/code tensors on GPU;
            # otherwise CUDA dequantisation receives CPU pointers.
            for module in transformer.modules():
                weight=getattr(module,'weight',None)
                state=getattr(weight,'quant_state',None)
                if state is not None:state.to(torch.device('cuda'))
                extra=getattr(module,'quant_state',None)
                if extra is not None and extra is not state:extra.to(torch.device('cuda'))
        else:transformer.to('cuda')
        (ROOT/'prompt_cache').mkdir(exist_ok=True)
        print('FLUX loaded','group' if group_offload else 'resident',round(time.time()-start,2),'seconds',flush=True)

    @torch.inference_mode()
    def embeddings(self,prompt):
        key=hashlib.sha256((REVISION+'|128|'+prompt).encode()).hexdigest()
        path=ROOT/'prompt_cache'/(key+'.pt')
        if path.exists():return torch.load(path,map_location='cpu',weights_only=True)
        # Keep only one large component on GPU at a time, including during encoding.
        if not self.group:self.pipe.transformer.to('cpu')
        self.pipe.text_encoder.to('cuda');self.pipe.text_encoder_2.to('cuda')
        embeddings,pooled,_=self.pipe.encode_prompt(prompt=prompt,device=torch.device('cuda'),num_images_per_prompt=1,max_sequence_length=128)
        result={'embeddings':embeddings.cpu(),'pooled':pooled.cpu()}
        torch.save(result,path)
        self.pipe.text_encoder.to('cpu');self.pipe.text_encoder_2.to('cpu');torch.cuda.empty_cache()
        if not self.group:self.pipe.transformer.to('cuda')
        return result

    @torch.inference_mode()
    def generate(self,prompt,seeds,size=384,steps=4):
        e=self.embeddings(prompt);n=len(seeds)
        emb=e['embeddings'].repeat(n,1,1).to('cuda');pooled=e['pooled'].repeat(n,1).to('cuda')
        generators=[torch.Generator(device='cuda').manual_seed(int(seed)) for seed in seeds]
        torch.cuda.synchronize();start=time.perf_counter()
        result=self.pipe(prompt_embeds=emb,pooled_prompt_embeds=pooled,height=size,width=size,
                         guidance_scale=0.,num_inference_steps=steps,max_sequence_length=128,generator=generators).images
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        return result,elapsed
