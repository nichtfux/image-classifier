from pathlib import Path
import json,numpy as np,torch
from evaluate_next import load,qd,write,OUT
from colour_data import ColourData
torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
net,_=load("colour299_40")
r,top,norm=qd(net,ColourData("test","colour299-40"),"painted",20261112)
previous=np.load(OUT/"colour299_40-next-painted-predictions.npz")["top3"]
print("frozen paint exact same top3 rows",float(np.all(top==previous,axis=1).mean()),flush=True)
p=Path("../notes/training-round-next.json");j=json.loads(p.read_text());j["colour299_40"]["quickdraw"]["painted"]=r;write(p,j)
np.savez(OUT/"colour299_40-next-painted-predictions.npz",top3=top,norm=norm)
print("DONE",r["real"],flush=True)
