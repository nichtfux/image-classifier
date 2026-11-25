import torch,bitsandbytes as bnb
x=bnb.nn.Linear4bit(256,256,bias=False,compute_dtype=torch.bfloat16,quant_type="nf4")
x=x.to("cpu")
print("CPU QUANT",x.weight.dtype,x.weight.shape,x.weight.quant_state is not None,flush=True)
x.cuda();print("TRANSFER",x(torch.ones(1,256,device="cuda",dtype=torch.bfloat16)).shape,flush=True)
