"""Freeze real-class features and outputs; learn safer custom boundaries.

Only recognized training data and our own train-split generation are embedded.
RGB attenuation is a model parameter, never a label-dependent serving rule.
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from balanced_validation import ROUND, OUT, write
from colour import NET
from colour_data import ColourData, inputs
from style_augment import augment
from train_next import load, save


def embeddings(net, x):
    values = []
    with torch.inference_mode():
        for part in x.split(64):
            if getattr(net, 'embedding_kind', None) == 'conv':
                h = part
                for index in (1, 2, 3):
                    h = F.max_pool2d(F.relu(getattr(net, f'b{index}')(getattr(net, f'c{index}')(h))), 2)
                values.append(h.flatten(1).float().cpu().numpy().astype(np.float16))
                continue
            h = net.features(part).float()
            scale = getattr(net, 'embedding_scale', None)
            h = h / scale if scale else F.normalize(h, dim=1)
            values.append(h.cpu().numpy().astype(np.float16))
    return np.concatenate(values)


def prepare_features(net, name):
    path = OUT / f'{name}-train-features.npz'
    if path.exists():
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}
    rng = np.random.default_rng(2026111103)
    data = ColourData('train', 'colour299-40')
    limit = 2000000
    first = np.asarray(data.y[:limit])
    chosen = np.sort(np.concatenate([rng.choice(np.flatnonzero(first == ci), 1000 if ci >= 270 else 320, replace=False)
                                     for ci in range(299)]))
    features, labels, domains = [], [], []
    palette = torch.as_tensor(NET, device='cuda')

    def append(x, y, domain):
        features.append(embeddings(net, x))
        labels.append(np.asarray(y, np.int64))
        domains.append(np.broadcast_to(domain, len(y)).copy())

    for lo in range(0, limit, 8192):
        hi = min(lo + 8192, limit)
        ix = chosen[(chosen >= lo) & (chosen < hi)] - lo
        if not len(ix):
            continue
        chunk = data.chunk(lo, hi, seed=2026111103 + lo, mode=int(lo % 5 == 0), fill=True)
        indices = torch.from_numpy(ix).cuda()
        x, y = inputs(chunk, indices)
        truth = y.cpu().numpy()
        black = torch.zeros_like(x)
        black[:, 0] = x[:, 0]
        append(black, truth, np.where(truth < 270, 0, 3))
        # Augment bounded subbatches, so preparation fits beside FLUX.
        for start in range(0, len(x), 128):
            selected = indices[start:start+128]
            sx = augment(x[start:start+128], chunk[6][selected], palette[chunk[5][selected].long()])
            target = truth[start:start+128]
            append(sx, target, np.where(target < 270, 0, 3))
        del chunk, x, black
    for bank, domain in [('qd-game', 1), ('generated', 2), ('procedural', 4)]:
        folder = getattr(net, 'generated_folder', 'head-generated') if bank == 'generated' else bank
        source_path = ROUND / folder / 'train.npz' if bank != 'procedural' else OUT.parent.parent / 'data/synth-diverse40/train.npz'
        with np.load(source_path) as source:
            x, truth = source['x'], source['y'].astype(np.int64)
        for lo in range(0, len(truth), 128):
            batch = torch.from_numpy(x[lo:lo+128].astype(np.float32) / 255).cuda().contiguous(memory_format=torch.channels_last)
            y = truth[lo:lo+128]
            group = np.where(y < 270, 2, 5) if bank == 'generated' else np.full(len(y), domain)
            append(batch, y, group)
            if bank != 'procedural':
                batch[:, 1:] = 0
                append(batch, y, group)
        print('Embedded', name, bank, len(truth), flush=True)
    result = {'h': np.concatenate(features), 'y': np.concatenate(labels),
              'domain': np.concatenate(domains).astype(np.uint8)}
    np.savez(path, **result)
    print('Embedded', name, 'total', sum(map(len, labels)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--rgb', type=float, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--free-scale', action='store_true')
    parser.add_argument('--reuse-features')
    parser.add_argument('--raw-features', action='store_true')
    parser.add_argument('--raw-scale', type=float, default=8.)
    parser.add_argument('--generated-folder', default='head-generated')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026111103)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.10)
    started = time.time()
    net, _ = load(OUT / 'gacl299_40_continued.pt')
    net.requires_grad_(False)
    net.generated_folder = args.generated_folder
    if args.raw_features:
        net.embedding_scale = args.raw_scale
    with torch.no_grad():
        net.c1.weight[:, 1:] *= args.rgb
    net.eval()
    bank = prepare_features(net, args.reuse_features or args.name)
    rng = np.random.default_rng(2026111103)
    groups = [np.flatnonzero(bank['domain'] == i) for i in range(6)]
    probabilities = []
    for indices in groups:
        counts = np.bincount(bank['y'][indices], minlength=299)
        p = 1 / np.sqrt(counts[bank['y'][indices]])
        probabilities.append(p / p.sum())
    net.f2.weight.requires_grad_(True)
    net.f2.bias.requires_grad_(True)
    if args.free_scale:
        with torch.no_grad():
            net.f2.weight.copy_(64 * F.normalize(net.f2.weight, dim=1))
        net.quantized_head = True
    net.f2.weight.register_hook(lambda gradient: torch.cat([torch.zeros_like(gradient[:270]), gradient[270:]]))
    net.f2.bias.register_hook(lambda gradient: torch.cat([torch.zeros_like(gradient[:270]), gradient[270:]]))
    original_real = net.f2.weight[:270].detach().clone()
    optimizer = torch.optim.Adam([
        {'params': [net.f2.weight], 'lr': .01 if args.free_scale else .00015},
        {'params': [net.f2.bias], 'lr': .01 if args.free_scale else .001},
    ])
    history = []
    for step in range(1, args.steps + 1):
        # 88% real-class negatives. No singleton receives inverse-frequency amplification.
        ix = np.concatenate([rng.choice(indices, n, p=p) for indices, p, n in zip(groups, probabilities, [328, 80, 40, 24, 24, 16])])
        h = torch.from_numpy(bank['h'][ix].astype(np.float32)).cuda()
        y = torch.from_numpy(bank['y'][ix]).cuda()
        weights = net.f2.weight.float() if args.free_scale else 64 * F.normalize(net.f2.weight.float(), dim=1)
        logits = F.linear(h, weights, net.f2.bias.float())
        real = y < 270
        real_score = logits[:, :270].max(1).values
        custom_score = logits[:, 270:].max(1).values
        correct_score = logits.gather(1, y[:, None])[:, 0]
        negative = (F.softplus(custom_score - real_score + 3) * real).mean()
        positive = (F.softplus(real_score - correct_score + 8) * ~real).mean()
        ce = F.cross_entropy(logits, y)
        loss = ce + 4 * negative + 2 * positive
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.f2.parameters(), 5.)
        optimizer.step()
        if step % 500 == 0:
            row = {'step': step, 'loss': float(loss.detach()), 'ce': float(ce.detach()),
                   'negative': float(negative.detach()), 'positive': float(positive.detach()),
                   'seconds': time.time() - started}
            history.append(row)
            print(args.name, row, flush=True)
        if step in (2000, args.steps):
            assert torch.equal(original_real, net.f2.weight[:270])
            exported = net
            if args.raw_features:
                exported = copy.deepcopy(net)
                with torch.no_grad():
                    exported.f2.weight.div_(args.raw_scale)
                exported.cosine = False
            save(exported, OUT / f'{args.name}_s{step}.pt', {
                'phase': 'balanced-v4-frozen-real', 'rgb_strength': args.rgb,
                'raw_feature_scale': args.raw_scale if args.raw_features else None,
                'source': 'gacl299_40_continued', 'steps': step, 'quality_tier': 'parked',
                'generated_snapshot': str(ROUND / args.generated_folder), 'history': history,
            })
            if exported is not net:
                del exported
    write(OUT / f'{args.name}-training.json', {'history': history, 'seconds': time.time() - started})


if __name__ == '__main__':
    main()
