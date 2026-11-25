import json,torch,numpy as np
from train_next import load,validation,OUT
from colour_data import ColourData
from gacl import quantize,from_quant
from evaluate_next import procedural
torch.set_num_threads(3);torch.backends.cudnn.benchmark=True
net,ck=load(OUT/"gacl299_40_wide-last.pt");q=quantize(net);qn=from_quant(q,cosine=net.cosine,norm_scale=float(net.norm_scale));del net
v=ColourData("validation","colour299-40");y=np.asarray(v.y);rows=[]
for ci in range(299):
 ix=np.flatnonzero(y==ci);rows.append((int(ix[0]),int(ix[0])+100))
r=validation(qn,v,rows);r["diverse"]=procedural(qn,"validation")
print(json.dumps({"checkpoint_phase":ck.get("phase"),"seen":ck.get("seen"),"metrics":r}),flush=True)
