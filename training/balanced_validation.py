"""Validation-only selection and calibration for the balanced v4 round.

The first 500 validation examples per class calibrate custom-class offsets.
The remaining 500 independently guard acceptance. No test data are read here.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from colour import NET
from colour_data import ColourData, inputs, cohort_metrics
from fill import fill_planes
from style_eval import augment
from gacl import quantize, from_quant
from train_next import load, OUT

ROOT = OUT.parent.parent
ROUND = Path(os.environ.get('DRAW_TRAIN_ROUND', str(ROOT / 'data/generated/rounds/balanced-v4-20261110')))
if not ROUND.is_absolute():
    ROUND = ROOT / ROUND
MODES = [('black', 100), ('policy', 20261110), ('random', 20261111),
         ('filled', 20261110), ('painted', 20261112)]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8', newline='\n')


def compact(net, x):
    values = []
    with torch.inference_mode():
        for batch in x.split(64):
            z = net(batch).float()
            real, ids = z[:, :270].topk(3, 1)
            values.append(torch.cat([real, ids.float(), z[:, 270:]], 1).cpu().numpy())
    return np.concatenate(values)


def cache(net, name):
    destination = OUT / f'{name}-validation'
    destination.mkdir(exist_ok=True)
    data = ColourData('validation', 'colour299-40')
    for mode, seed in MODES:
        path = destination / f'{mode}.npz'
        if path.exists():
            continue
        painted = mode in ('filled', 'painted')
        colours = None if mode == 'black' else data.fixed('policy' if painted else mode, seed)
        fills = data.fixed_fills(colours, seed + 300000) if painted else None
        logits = []
        # Keep the canonical 512-item augmentation RNG sequence, independent of
        # the smaller forward microbatches needed alongside background FLUX.
        with torch.inference_mode():
            for lo in range(0, len(data.y), 512):
                hi = min(lo + 512, len(data.y))
                chunk = data.chunk(lo, hi, device='cuda', fixed=colours,
                                   black=mode == 'black', fill=painted, fixed_fill=fills)
                x, _ = inputs(chunk, black=mode == 'black')
                if painted:
                    generator = torch.Generator(device='cuda').manual_seed(seed + lo)
                    rgb = torch.as_tensor(NET, device='cuda')[chunk[5].long()]
                    if mode == 'filled':
                        fill_planes(x, rgb, torch.ones(len(x), device='cuda', dtype=torch.bool),
                                    generator=generator, region=chunk[6])
                    else:
                        x = augment(x, chunk[6], rgb, generator=generator)
                logits.append(compact(net, x))
        truth = np.asarray(data.y)
        role = np.tile(np.r_[np.zeros(500), np.ones(500)], 299).astype(np.uint8)
        np.savez(path, z=np.concatenate(logits), y=truth, role=role)
        print('Cached validation', name, mode, flush=True)
    for bank in ['qd-game', 'generated']:
        with np.load(ROUND / bank / 'validation.npz') as source:
            x, truth = source['x'], source['y']
            if bank == 'qd-game':
                role = source['role']
            else:
                import hashlib
                from generated_splits import group_key
                manifest = json.loads((ROUND / 'generated/manifest.json').read_text())
                groups = {row['id']: group_key(row) for row in manifest['source_rows']}
                role = np.asarray([int(hashlib.sha256(groups[str(i)].encode()).hexdigest()[:8], 16) % 2
                                   for i in source['ids']], np.uint8)
        for black in (False, True):
            key = bank + ('-black' if black else '')
            path = destination / f'{key}.npz'
            if path.exists():
                continue
            values = []
            for lo in range(0, len(truth), 64):
                batch = torch.from_numpy(x[lo:lo+64].astype(np.float32) / 255).cuda()
                if black:
                    batch[:, 1:] = 0
                values.append(compact(net, batch.contiguous(memory_format=torch.channels_last)))
            np.savez(path, z=np.concatenate(values), y=truth, role=role)
    return destination


def top3(z, custom_offset=0., hole_offset=0.):
    custom = z[:, 6:].copy() - custom_offset
    custom[:, 5] -= hole_offset
    values = np.concatenate([z[:, :3], custom], 1)
    ids = np.concatenate([z[:, 3:6].astype(np.int16),
                          np.broadcast_to(np.arange(270, 299), custom.shape)], 1)
    order = np.argsort(-values, axis=1)[:, :3]
    return np.take_along_axis(ids, order, 1)


def cohort(z, y, custom_offset=0., hole_offset=0.):
    top = top3(z, custom_offset, hole_offset)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = cohort_metrics(y, top)
    for group in ('all', 'real', 'synthetic'):
        if result[group]['drawings'] == 0:
            result[group]['top1'] = None
            result[group]['top3'] = None
    real = y < 270
    result['black_hole_rate'] = float(np.mean(top[real, 0] == 275)) if real.any() else 0.
    return result


def required_offset(margin, allowed):
    if len(margin) <= allowed:
        return 0.
    boundary = np.partition(margin, len(margin) - allowed - 1)[len(margin) - allowed - 1]
    return max(0., float(boundary) + .001)


def select(name):
    if (ROUND / 'test-evaluation-started.json').exists():
        raise RuntimeError('This round is frozen and tests have opened. Use a new round for selection.')
    baseline = OUT / 'colour299_40-validation'
    candidate = OUT / f'{name}-validation'
    shared_offset = 0.
    black_hole_offset = 0.
    keys = [p.stem for p in sorted(baseline.glob('*.npz'))]
    for key in keys:
        with np.load(baseline / f'{key}.npz') as b, np.load(candidate / f'{key}.npz') as c:
            use = (c['role'] == 0) & (c['y'] < 270)
            if not use.any():
                continue
            bz, cz = b['z'][use], c['z'][use]
            bt = top3(bz)[:, 0]
            # Fixed twofold safety margin, chosen before examining these scores.
            allowed = int(((bt >= 270) & (bt != 275)).sum())
            allowed_hole = int((bt == 275).sum())
            other_custom = np.delete(cz[:, 6:], 5, axis=1)
            shared_offset = max(shared_offset, required_offset(other_custom.max(1) - cz[:, 0], allowed))
            black_hole_offset = max(black_hole_offset, required_offset(cz[:, 11] - cz[:, 0], allowed_hole))
    custom_offsets = np.full(29, shared_offset, np.float32)
    custom_offsets[5] = max(shared_offset, black_hole_offset)
    from constrained_bias import fit_offsets
    constrained, calibration_info = fit_offsets(baseline, candidate, keys, top3)
    if constrained is not None:
        custom_offsets = constrained
    result = {'name': name, 'custom_offsets': custom_offsets.tolist(),
              'calibration_rule': 'Recall-constrained per-output offsets on calibration halves, conservative union-bound FA budgets, independent validation guards',
              'calibration_solver': calibration_info,
              'guards': {}, 'baseline': {}, 'candidate': {}}
    for key in keys:
        with np.load(baseline / f'{key}.npz') as b, np.load(candidate / f'{key}.npz') as c:
            use = c['role'] == 1
            bc = cohort(b['z'][use], b['y'][use])
            cc = cohort(c['z'][use], c['y'][use], custom_offsets)
        result['baseline'][key] = bc
        result['candidate'][key] = cc
        if cc['real']['drawings']:
            result['guards'][key + '-custom-alarms'] = cc['false_alarms']['rate'] <= bc['false_alarms']['rate']
            result['guards'][key + '-black-hole'] = cc['black_hole_rate'] <= bc['black_hole_rate']
        if key == 'black':
            for metric in ['top1', 'top3']:
                result['guards']['black-real-' + metric] = cc['real'][metric] >= bc['real'][metric]
                result['guards']['black-synth-' + metric] = cc['synthetic'][metric] >= bc['synthetic'][metric] - .002
        if key in ('filled', 'painted'):
            result['guards'][key + '-improves'] = cc['real']['top1'] >= bc['real']['top1'] + .01
            for metric in ('top1', 'top3'):
                result['guards'][key + '-custom-' + metric] = cc['synthetic'][metric] >= bc['synthetic'][metric] - .002
        if key in ('policy', 'random'):
            result['guards'][key + '-preserved'] = cc['real']['top1'] >= bc['real']['top1'] - .002
    result['eligible'] = all(result['guards'].values())
    if 'headr0_' in name:
        result['guards']['colour-retained'] = False
        result['eligible'] = False
    result['rank'] = sum(result['candidate'][k]['real']['top1'] for k in ('filled', 'painted')) / 2
    write(OUT / f'{name}-selection.json', result)
    print('VALIDATION', name, 'eligible', result['eligible'], 'offsets', custom_offsets.tolist(),
          'failed', [key for key, passed in result['guards'].items() if not passed], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.10)
    if not args.select_only:
        if args.name == 'colour299_40':
            from evaluate_next import load as load_export
            net, _ = load_export(args.name)
        else:
            original, _ = load(OUT / f'{args.name}.pt')
            net = from_quant(quantize(original), cosine=original.cosine)
            del original
        cache(net, args.name)
    if args.name != 'colour299_40':
        select(args.name)


if __name__ == '__main__':
    main()
