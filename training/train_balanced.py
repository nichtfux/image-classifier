"""Balanced conversion-v4 retraining, with frozen validation-only guards."""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from colour import NET
from colour_data import ColourData, inputs
from style_augment import augment
from train_next import load, save, OUT
from balanced_validation import ROUND, write


class Bank:
    def __init__(self, path, family=None):
        with np.load(path) as data:
            keep = np.ones(len(data['y']), bool)
            if family == 'real':
                keep = data['y'] < 270
            elif family == 'custom':
                keep = data['y'] >= 270
            self.x = data['x'][keep]
            self.y = data['y'][keep]
            self.region = data['region'][keep] if 'region' in data else None
            self.fill = data['fill'][keep] if 'fill' in data else None
        counts = np.bincount(self.y, minlength=299)
        # Square-root class balancing avoids repeatedly memorizing singleton styles.
        weight = 1 / np.sqrt(counts[self.y])
        self.prob = weight / weight.sum()

    def draw(self, number, rng):
        ix = rng.choice(len(self.y), number, p=self.prob)
        x = torch.from_numpy(self.x[ix].astype(np.float32) / 255).cuda()
        y = torch.from_numpy(self.y[ix].astype(np.int64)).cuda()
        return x.contiguous(memory_format=torch.channels_last), y, ix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--negative', type=float, required=True)
    parser.add_argument('--steps', type=int, default=4500)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026111102)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.10)
    rng = np.random.default_rng(2026111102)
    net, source = load(OUT / 'gacl299_40_continued.pt')
    from evaluate_next import load as load_export
    teacher, _ = load_export('colour299_40')
    net.train()
    for module in net.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.Dropout)):
            module.eval()
    # Learned output offsets are calibrated after quantization on a separate
    # validation half. Keep them neutral while learning class features.
    net.f2.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in net.named_parameters() if not n.startswith(('f1', 'f2'))], 'lr': .000012},
        {'params': [p for n, p in net.named_parameters() if n.startswith(('f1', 'f2')) and p.requires_grad], 'lr': .00004},
    ], weight_decay=.0001)
    banks = [Bank(ROUND / 'qd-game/train.npz'), Bank(ROUND / 'generated/train.npz', 'real'),
             Bank(ROUND / 'generated/train.npz', 'custom'), Bank(OUT.parent.parent / 'data/synth-diverse40/train.npz')]
    data = ColourData('train', 'colour299-40')
    palette = torch.as_tensor(NET, device='cuda')
    started = time.time()
    step = 0
    history = []
    while step < args.steps:
        lo = int(rng.integers(0, (len(data.y) - 10000) // 256)) * 256
        chunk = data.chunk(lo, lo + 10000, seed=2026111102 + step,
                           mode=int((step // 156) % 5 == 0), fill=True)
        order = torch.randperm(10000, device='cuda')
        for pos in range(0, len(order) - 64, 64):
            if step >= args.steps:
                break
            ix = order[pos:pos + 64]
            x, y = inputs(chunk, ix)
            anchors = torch.zeros_like(x[:16])
            anchors[:, 0] = x[:16, 0]
            anchor_y = y[:16].clone()
            x = augment(x, chunk[6][ix], palette[chunk[5][ix].long()])
            # 44 QD, 8 game-converted QD real negatives, 6 generated real,
            # 2 generated custom and 4 independently programmed custom styles.
            start = 44
            for bank, number in zip(banks, [8, 6, 2, 4]):
                bx, by, bi = bank.draw(number, rng)
                if bank.region is not None:
                    region = torch.from_numpy(bank.region[bi].astype(bool)).cuda()
                    fill = palette[torch.from_numpy(bank.fill[bi].astype(np.int64)).cuda()]
                    bx = augment(bx, region, fill)
                else:
                    black = torch.rand(number, device='cuda') < .4
                    bx[black, 1:] = 0
                x[start:start + number] = bx
                y[start:start + number] = by
                start += number
            real = y < 270
            weights = torch.ones(64, device='cuda')
            weights[44:52] = .5
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = net(x).float()
                with torch.no_grad():
                    target = teacher(x).float()
                    anchor_target = teacher(anchors).float()
                ce = (F.cross_entropy(logits, y, reduction='none') * weights).mean()
                kl = F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(target / 2, 1), reduction='none').sum(1) * 4
                retention = (kl * real).mean() * .5
                real_score = logits[:, :270].max(1).values
                custom_score = logits[:, 270:].max(1).values
                family_penalty = (F.softplus(custom_score - real_score + 3) * real).mean()
                hole_penalty = (F.softplus(logits[:, 275] - real_score + 5) * real).mean()
                az = net(anchors).float()
                anchor_loss = F.cross_entropy(az, anchor_y)
                anchor_kl = F.kl_div(F.log_softmax(az / 2, 1), F.softmax(anchor_target / 2, 1), reduction='batchmean') * 4
                loss = ce + retention + args.negative * (family_penalty + hole_penalty) + .25 * (anchor_loss + .5 * anchor_kl)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            step += 1
            fraction = step / args.steps
            factor = .3 + .7 * .5 * (1 + math.cos(math.pi * fraction))
            optimizer.param_groups[0]['lr'] = .000012 * factor
            optimizer.param_groups[1]['lr'] = .00004 * factor
            if step % 100 == 0:
                row = {'step': step, 'seconds': time.time() - started, 'loss': float(loss.detach()),
                       'ce': float(ce.detach()), 'negative': float((family_penalty + hole_penalty).detach())}
                history.append(row)
                print(args.name, json.dumps(row), flush=True)
            if step in (1500, args.steps):
                name = args.name + f'_s{step}'
                save(net, OUT / f'{name}.pt', {
                    'phase': 'balanced-v4', 'source': 'gacl299_40_continued', 'steps': step,
                    'negative_weight': args.negative, 'generated_snapshot': str(ROUND),
                    'quality_tier': 'parked', 'history': history,
                })
        del chunk
    write(OUT / f'{args.name}-training.json', {'history': history, 'seconds': time.time() - started})


if __name__ == '__main__':
    main()
