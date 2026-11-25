"""Counterfactual all-black replay of the same captured client stroke geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from colour import planes
from colour_raster import reference
from train_colour import quant_net

ROOT = Path(__file__).resolve().parent.parent


def main():
    torch.set_num_threads(4)
    out = ROOT / 'training/out'
    labels = json.loads((out/'colour299_40.json').read_text())['categories']
    with np.load(out/'colour299_40.npz') as z:
        weights = [(z[f'w{i}'],z[f's{i}'],z[f'b{i}']) for i in range(5)]
    net = quant_net(weights,'cuda')
    records, images = [], []
    for path in sorted((ROOT/'notes').glob('live-paced-after-*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        strokes = row['capture']['strokes']
        raw = [[[p[0] for p in s['points']],[p[1] for p in s['points']]] for s in strokes]
        ink, owner = reference(raw,40)
        x = planes(ink,owner,[s['colour']-1 for s in strokes])
        black = x.copy()
        black[1:] = 0
        images.extend((x,black))
        records.append({'file':path.name,'label':row['case']['label'],
                        'paint':row['case']['paint'],'nominal_scale':row['case']['size']})
    with torch.no_grad():
        probs = net(torch.from_numpy(np.stack(images)).float().cuda().contiguous(memory_format=torch.channels_last)).softmax(1).cpu().numpy()
    for i,row in enumerate(records):
        for mode,values in [('original_colour',probs[2*i]),('same_ink_all_black',probs[2*i+1])]:
            ids = np.argsort(-values)[:3]
            row[mode] = [{'label':labels[int(j)],'prob':float(values[j])} for j in ids]
    result = {'model':'colour299_40','caveat':'Same twelve authored development cases. RGB input removed without changing any ink pixels. Not a real-player accuracy estimate. No training.', 'records':records}
    (ROOT/'notes/live-colour-ablation.json').write_text(json.dumps(result,indent=1)+'\n',encoding='utf-8',newline='\n')
    for row in records:
        print(row['label'],row['nominal_scale'],'paint',row['paint'],row['original_colour'][0]['label'],'-> black',row['same_ink_all_black'][0]['label'])


if __name__ == '__main__':
    main()
