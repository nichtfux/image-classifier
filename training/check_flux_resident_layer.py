import torch
from diffusers import FluxTransformer2DModel
m=FluxTransformer2DModel.from_pretrained("../data/generated/models/flux1-schnell-nf4/transformer",local_files_only=True,torch_dtype=torch.bfloat16,device_map="cpu")
xm=m.x_embedder.cuda();print("X EMBEDDER",type(xm).__name__,flush=True)
y=xm(torch.randn(1,256,64,device="cuda",dtype=torch.bfloat16));torch.cuda.synchronize();print("RESIDENT LAYER",y.shape,torch.isfinite(y).all().item(),flush=True)
