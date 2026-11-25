from pathlib import Path
import json,numpy as np
from evaluate_next import quality,OUT,write
p=Path("../notes/training-round-next.json");j=json.loads(p.read_text())
qn=np.load(OUT/"colour299_40-next-black-predictions.npz")["norm"]
tn=np.load(OUT/"colour299_40-next-tu-line.npz")["norm"]
r,t,q=quality(None,qn,tn);r["trained_with_gacl"]=False
j["colour299_40"]["quality"]=r;write(p,j)
np.savez(OUT/"colour299_40-quality-comparison.npz",tu=t,quickdraw=q)
print({k:v for k,v in r.items() if k!="classes"},flush=True)
