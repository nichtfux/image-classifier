"""Evaluate r4 on the same fixed-seed colour/fill presentations used for both sizes."""
import json
from pathlib import Path
import numpy as np
import torch
from colour_data import ColourData,cohort_metrics
from train_colour import evaluate,quant_net

if __name__=='__main__':
    torch.set_num_threads(6);here=Path(__file__).resolve().parent;out=here/'out'
    with np.load(out/'wide299_28_r4.npz') as z:q=[(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    with np.load(out/'wide299_28_r4.predictions.npz') as z:black=cohort_metrics(z['truth'],z['top3'])
    result={m:black for m in ('black','policy','random')}
    result['filled']=evaluate(quant_net(q,'cuda'),ColourData('test'),'filled',20261112,'cuda',save=out/'r4.filled.predictions.npz')
    (out/'r4.colour-baseline.json').write_text(json.dumps(result,indent=1),encoding='utf-8',newline='\n')
    print(json.dumps(result),flush=True)
