"""Add retained developer-capture replays to already-completed offline reports."""
import json,torch
from evaluate_next import ROOT,load,write,captured_replays

if __name__=='__main__':
    torch.set_num_threads(3)
    path=ROOT/'notes/training-round-next.json';report=json.loads(path.read_text())
    for name in ['colour299_40','gacl299_40_wide']:
        net,_=load(name,'cpu');result=captured_replays(net)
        report[name]['captured_replays']=result
        print(name,result['correct'],'/',result['drawings'],flush=True)
        for row in result['records']:print(row['label'],row['paint'],row['nominal_scale'],'->',row['top3'][0]['label'],flush=True)
    write(path,report)
