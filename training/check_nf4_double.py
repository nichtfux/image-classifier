import torch,bitsandbytes as bnb
for compressed in [False,True]:
 m=bnb.nn.Linear4bit(1024,4096,bias=False,compute_dtype=torch.bfloat16,quant_type="nf4",compress_statistics=compressed).to(dtype=torch.bfloat16).to("cpu")
 m.cuda();y=m(torch.randn(2,128,1024,device="cuda",dtype=torch.bfloat16));torch.cuda.synchronize();print("LARGE NF4",compressed,y.shape,torch.isfinite(y).all().item(),flush=True)
