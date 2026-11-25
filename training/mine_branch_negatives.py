"""Mine custom confusions from recognized QD TRAIN drawings only."""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from balanced_validation import OUT, write
from colour import NET
from colour_data import ColourData, inputs
from fill import fill_planes
from style_augment import augment
from style_eval import augment as stress_augment
from train_next import load


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.10)
    net, _ = load(OUT / 'balanced299_40_branch768_s6000.pt')
    net.requires_grad_(False)
    data = ColourData('train', 'colour299-40')
    palette = torch.as_tensor(NET, device='cuda')
    rows = []
    labels = []
    source_ids = []
    predictions = []
    margins = []
    kinds = []
    started = time.time()
    first, last = 5000000, 6000000
    seen = 0
    with torch.inference_mode():
        for batch, lo in enumerate(range(first, last, 512)):
            hi = min(lo + 512, last)
            chunk = data.chunk(lo, hi, seed=2026111200 + lo, mode=int(batch % 5 == 0), fill=True)
            x, y = inputs(chunk)
            kind = batch % 5
            generator = torch.Generator(device='cuda').manual_seed(2026111200 + lo)
            if kind == 0:
                x[:, 1:] = 0
            elif kind == 1:
                fill_planes(x, palette[chunk[5].long()], torch.ones(len(x), device='cuda', dtype=torch.bool),
                            generator=generator, region=chunk[6])
            else:
                operation = stress_augment if kind == 2 else augment
                x = operation(x, chunk[6], palette[chunk[5].long()], generator=generator)
            for start in range(0, len(x), 64):
                image = x[start:start+64]
                truth = y[start:start+64]
                h = image
                for index in (1, 2, 3):
                    h = F.max_pool2d(F.relu(getattr(net, f'b{index}')(getattr(net, f'c{index}')(h))), 2)
                h = h.flatten(1)
                logits = net.f2(F.relu(net.f1(h)))
                custom, target = logits[:, 270:].max(1)
                margin = custom - logits[:, :270].max(1).values
                use = (truth < 270) & (margin > -4)
                if use.any():
                    rows.append(h[use].cpu().numpy().astype(np.float16))
                    labels.append(truth[use].cpu().numpy())
                    source_ids.append(np.arange(lo+start, lo+start+len(image))[use.cpu().numpy()])
                    predictions.append(target[use].cpu().numpy())
                    margins.append(margin[use].cpu().numpy())
                    kinds.append(np.full(int(use.sum()), kind, np.uint8))
            seen += int((y < 270).sum())
            if batch % 250 == 0:
                print('MINING TRAIN', lo, 'real seen', seen, 'kept', sum(map(len, labels)),
                      'seconds', round(time.time() - started), flush=True)
    h, y = np.concatenate(rows), np.concatenate(labels)
    prediction, margin = np.concatenate(predictions), np.concatenate(margins)
    # Keep the strongest cases per confused output, bounding memory while
    # retaining the dominant failure classes and uncommon custom destinations.
    chosen = np.concatenate([ix[np.argsort(-margin[ix])[:4000]]
                             for label in range(29) if len(ix := np.flatnonzero(prediction == label))])
    np.savez(OUT / 'branch-hard-negatives.npz', h=h[chosen], y=y[chosen],
             source_ids=np.concatenate(source_ids)[chosen], custom_prediction=prediction[chosen] + 270,
             margin=margin[chosen], style=np.concatenate(kinds)[chosen])
    write(OUT / 'branch-hard-negatives.json', {
        'split': 'train', 'source_range': [first, last], 'recognized_real_seen': seen,
        'near_custom_examples': len(y), 'retained': len(chosen), 'minimum_margin': -4,
        'checkpoint': 'balanced299_40_branch768_s6000.pt',
        'styles': ['black', 'filled', 'frozen brush recipe, training seed', 'current brush recipe', 'current brush recipe'],
        'seconds': time.time() - started,
    })
    print('MINING COMPLETE', len(chosen), flush=True)


if __name__ == '__main__':
    main()
