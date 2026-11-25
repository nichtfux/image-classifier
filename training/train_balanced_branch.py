"""Extra custom residual features preserve the original real classifier.

The first 384 hidden units and their 299 original outputs stay frozen. Another
384 units learn custom corrections, regularized to zero on old synthetic data.
The resulting network still exports as the existing three-conv/two-dense graph.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from gacl import QualityNet
from train_next import load, save, OUT
from train_balanced_head import prepare_features
from balanced_validation import ROUND
from colour_data import ColourData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='balanced299_40_branch768')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--resume')
    parser.add_argument('--reuse-features')
    parser.add_argument('--hard-bank')
    parser.add_argument('--retention', type=float, default=1.)
    parser.add_argument('--old-margin-weight', type=float, default=0.)
    parser.add_argument('--fresh-old-black', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026111105)
    torch.cuda.set_per_process_memory_fraction(.10)
    source, _ = load(OUT / 'gacl299_40_continued.pt')
    source.requires_grad_(False)
    with torch.no_grad():
        source.c1.weight[:, 1:] *= .5
    source.embedding_kind = 'conv'
    source.generated_folder = 'raw-generated'
    bank = prepare_features(source, args.reuse_features or args.name)
    net = QualityNet((48, 96, 192), 768).cuda().eval()
    state = source.state_dict()
    for key in list(state):
        if not key.startswith(('f1.', 'f2.')):
            net.state_dict()[key].copy_(state[key])
    with torch.no_grad():
        net.f1.weight[:384].copy_(source.f1.weight)
        net.f1.bias[:384].copy_(source.f1.bias)
        net.f1.weight[384:].copy_(source.f1.weight)
        net.f1.weight[384:].add_(torch.randn_like(source.f1.weight) * .0001)
        net.f1.bias[384:].copy_(source.f1.bias)
        net.f2.weight.zero_()
        net.f2.weight[:, :384].copy_(64 * F.normalize(source.f2.weight, dim=1) / 8)
        net.f2.bias.copy_(source.f2.bias)
    if args.resume:
        net, _ = load(OUT / args.resume)
    net.requires_grad_(False)
    for parameter in net.f1.parameters():
        parameter.requires_grad_(True)
    net.f2.weight.requires_grad_(True)
    net.f2.bias.requires_grad_(True)
    net.f1.weight.register_hook(lambda gradient: torch.cat([torch.zeros_like(gradient[:384]), gradient[384:]]))
    net.f1.bias.register_hook(lambda gradient: torch.cat([torch.zeros_like(gradient[:384]), gradient[384:]]))
    def custom_gradient(gradient):
        result = torch.zeros_like(gradient)
        result[270:, 384:] = gradient[270:, 384:]
        return result
    net.f2.weight.register_hook(custom_gradient)
    net.f2.bias.register_hook(lambda gradient: torch.cat([torch.zeros_like(gradient[:270]), gradient[270:]]))
    real_weights = net.f2.weight[:270].detach().clone()
    old_features = net.f1.weight[:384].detach().clone()
    optimizer = torch.optim.Adam([
        {'params': list(net.f1.parameters()), 'lr': .00002},
        {'params': [net.f2.weight], 'lr': .00015},
        {'params': [net.f2.bias], 'lr': .001},
    ])
    groups = [np.flatnonzero(bank['domain'] == i) for i in range(6)]
    probabilities = []
    for indices in groups:
        counts = np.bincount(bank['y'][indices], minlength=299)
        p = 1 / np.sqrt(counts[bank['y'][indices]])
        probabilities.append(p / p.sum())
    rng = np.random.default_rng(2026111105)
    started = time.time()
    history = []
    hard = None
    if args.hard_bank:
        with np.load(OUT / args.hard_bank) as data:
            hard = {key: data[key] for key in ['h', 'y']}
    old_data = ColourData('train', 'colour299-40') if args.fresh_old_black else None
    old_ids = np.flatnonzero(old_data.y >= 270) if old_data is not None else None
    for step in range(1, args.steps + 1):
        ix = np.concatenate([rng.choice(indices, n, p=p) for indices, p, n in zip(groups, probabilities, [288, 80, 40, 64, 24, 16])])
        x = torch.from_numpy(bank['h'][ix].astype(np.float32)).cuda()
        y = torch.from_numpy(bank['y'][ix]).cuda()
        old_synthetic = torch.from_numpy(bank['domain'][ix] == 3).cuda()
        if hard is not None:
            selected = rng.integers(0, len(hard['y']), 192)
            x[:192] = torch.from_numpy(hard['h'][selected].astype(np.float32)).cuda()
            y[:192] = torch.from_numpy(hard['y'][selected]).cuda()
        if old_data is not None:
            selected = rng.choice(old_ids, 64)
            ink = torch.from_numpy(np.array(old_data.x[selected])).cuda().float() / 255
            image = torch.zeros((64, 4, 40, 40), device='cuda')
            image[:, 0] = ink
            with torch.no_grad():
                for index in (1, 2, 3):
                    image = F.max_pool2d(F.relu(getattr(net, f'b{index}')(getattr(net, f'c{index}')(image))), 2)
            x[408:472] = image.flatten(1)
            y[408:472] = torch.from_numpy(np.array(old_data.y[selected])).cuda()
        h = F.relu(net.f1(x))
        original = F.linear(h[:, :384], net.f2.weight[:, :384], source.f2.bias)
        residual = F.linear(h[:, 384:], net.f2.weight[270:, 384:], net.f2.bias[270:] - source.f2.bias[270:])
        logits = torch.cat([original[:, :270], original[:, 270:] + residual], 1)
        real = y < 270
        real_score = logits[:, :270].max(1).values
        true_score = logits.gather(1, y[:, None])[:, 0]
        negative = (F.softplus(logits[:, 270:].max(1).values - real_score + 3) * real).mean()
        positive = (F.softplus(real_score - true_score + 5) * ~real).mean()
        retention = (residual.square().mean(1) * old_synthetic).mean()
        rivals = torch.ones_like(logits, dtype=torch.bool).scatter_(1, y[:, None], False)
        original_margin = original.gather(1, y[:, None])[:, 0] - original.masked_fill(~rivals, -torch.inf).max(1).values
        current_margin = true_score - logits.masked_fill(~rivals, -torch.inf).max(1).values
        protected = old_synthetic & (original_margin > 0)
        margin_retention = (F.relu(original_margin.clamp(max=2) - current_margin).square() * protected).mean()
        loss = (F.cross_entropy(logits, y) + 4 * negative + positive
                + args.retention * retention + args.old_margin_weight * margin_retention)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.f1.parameters()) + list(net.f2.parameters()), 5.)
        optimizer.step()
        if step % 500 == 0:
            row = {'step': step, 'loss': float(loss.detach()), 'negative': float(negative.detach()),
                   'positive': float(positive.detach()), 'retention': float(retention.detach()),
                   'margin_retention': float(margin_retention.detach()),
                   'seconds': time.time() - started}
            history.append(row)
            print(args.name, row, flush=True)
        if step in (2000, args.steps):
            assert torch.equal(real_weights, net.f2.weight[:270])
            assert torch.equal(old_features, net.f1.weight[:384])
            save(net, OUT / f'{args.name}_s{step}.pt', {
                'phase': 'balanced-v4-residual-branch', 'rgb_strength': .5,
                'source': 'gacl299_40_continued', 'steps': step, 'quality_tier': 'parked',
                'generated_snapshot': str(ROUND / 'raw-generated'), 'history': history,
                'raw_feature_scale': 8.,
                'hard_negatives': args.hard_bank, 'retention_weight': args.retention,
                'old_margin_weight': args.old_margin_weight,
                'fresh_old_black': args.fresh_old_black,
            })


if __name__ == '__main__':
    main()
