"""Fit custom-output slopes/intercepts without forgetting their feature detectors."""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from train_next import load, save, OUT
from train_balanced_head import prepare_features
from balanced_validation import ROUND


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--rgb', type=float, required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--steps', type=int, default=6000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026111104)
    torch.cuda.set_per_process_memory_fraction(.10)
    net, _ = load(OUT / 'gacl299_40_continued.pt')
    net.requires_grad_(False)
    with torch.no_grad():
        net.c1.weight[:, 1:] *= args.rgb
        net.f2.weight.copy_(64 * F.normalize(net.f2.weight, dim=1))
    net.quantized_head = True
    original_weight = net.f2.weight.detach().clone()
    original_bias = net.f2.bias.detach().clone()
    bank = prepare_features(net, args.features)
    rng = np.random.default_rng(2026111104)
    groups = [np.flatnonzero(bank['domain'] == i) for i in range(6)]
    probabilities = []
    for indices in groups:
        counts = np.bincount(bank['y'][indices], minlength=299)
        p = 1 / np.sqrt(counts[bank['y'][indices]])
        probabilities.append(p / p.sum())
    raw_gain = torch.full((29,), math.log(math.expm1(1.)), device='cuda', requires_grad=True)
    bias = torch.zeros(29, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([raw_gain, bias], lr=.015)
    started = time.time()
    history = []
    for step in range(1, args.steps + 1):
        ix = np.concatenate([rng.choice(indices, n, p=p) for indices, p, n in zip(groups, probabilities, [328, 80, 40, 24, 24, 16])])
        h = torch.from_numpy(bank['h'][ix].astype(np.float32)).cuda()
        y = torch.from_numpy(bank['y'][ix]).cuda()
        with torch.no_grad():
            original_logits = F.linear(h, original_weight, original_bias)
        gain = F.softplus(raw_gain)
        custom_logits = original_logits[:, 270:] * gain + bias
        logits = torch.cat([original_logits[:, :270], custom_logits], 1)
        real = y < 270
        real_score = logits[:, :270].max(1).values
        custom_score = custom_logits.max(1).values
        true_score = logits.gather(1, y[:, None])[:, 0]
        negative = (F.softplus(custom_score - real_score + 3) * real).mean()
        positive = (F.softplus(real_score - true_score + 8) * ~real).mean()
        loss = F.cross_entropy(logits, y) + 4 * negative + 2 * positive
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            row = {'step': step, 'loss': float(loss.detach()), 'negative': float(negative.detach()),
                   'positive': float(positive.detach()), 'seconds': time.time() - started}
            history.append(row)
            print(args.name, row, flush=True)
        if step in (2000, args.steps):
            with torch.no_grad():
                net.f2.weight[270:].copy_(original_weight[270:] * F.softplus(raw_gain)[:, None])
                net.f2.bias[270:].copy_(original_bias[270:] * F.softplus(raw_gain) + bias)
            save(net, OUT / f'{args.name}_s{step}.pt', {
                'phase': 'balanced-v4-affine', 'rgb_strength': args.rgb, 'steps': step,
                'source': 'gacl299_40_continued', 'quality_tier': 'parked',
                'generated_snapshot': str(ROUND / 'head-generated'), 'history': history,
                'custom_gains': F.softplus(raw_gain).detach().cpu().tolist(),
                'custom_offsets': bias.detach().cpu().tolist(),
            })


if __name__ == '__main__':
    main()
