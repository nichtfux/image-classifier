"""One final evaluation after validation-only selection is frozen."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from balanced_validation import ROUND, MODES, write
from train_next import load, save, OUT
from gacl import quantize, from_quant
from colour_data import ColourData
from evaluate_next import qd, external, procedural, probes, captured_replays

ROOT = OUT.parent.parent


def limit_batches(net):
    original = net.forward

    def forward(x, details=False):
        result = [original(part, details=details) for part in x.split(64)]
        if details:
            return tuple(torch.cat([row[i] for row in result]) for i in range(2))
        return torch.cat(result)

    net.forward = forward
    return net


def main():
    selection = json.loads((ROUND / 'selection.json').read_text())
    selected = selection['selected']
    if selected is None:
        print('No validation-eligible candidate. Final tests remain unopened.', flush=True)
        return
    name = selected + '_calibrated'
    marker = ROUND / 'test-evaluation-started.json'
    if marker.exists():
        raise RuntimeError('Final evaluation already started. Inspect saved partial results before resuming.')
    chosen = next(row for row in selection['candidates'] if row['name'] == selected)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.10)
    original, checkpoint = load(OUT / f'{selected}.pt')
    with torch.no_grad():
        original.f2.bias[270:] -= torch.tensor(chosen['custom_offsets'], device='cuda')
    save(original, OUT / f'{name}.pt', {
        'phase': 'balanced-v4-calibrated', 'source': selected, 'quality_tier': 'parked',
        'calibration': {'custom_offsets': chosen['custom_offsets']},
        'rgb_strength': checkpoint.get('rgb_strength', 1.),
        'raw_feature_scale': checkpoint.get('raw_feature_scale'),
        'generated_snapshot': checkpoint.get('generated_snapshot', str(ROUND / 'generated')),
    })
    q = quantize(original)
    np.savez(OUT / f'{name}.npz', **{f'{k}{i}': v for i, row in enumerate(q) for k, v in zip(['w', 's', 'b'], row)})
    write(marker, {'name': name, 'checkpoint_sha256': hashlib.sha256((OUT / f'{name}.pt').read_bytes()).hexdigest(),
                   'weights_sha256': hashlib.sha256((OUT / f'{name}.npz').read_bytes()).hexdigest(),
                   'frozen_selection': selection['frozen_utc'], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    net = limit_batches(from_quant(q, cosine=original.cosine))
    del original
    from evaluate_next import load as load_export
    installed_net, _ = load_export('colour299_40')
    installed_net = limit_batches(installed_net)
    baseline = {'name': 'colour299_40', 'quickdraw': {}, 'external': {}}
    baseline_path = ROOT / 'notes/balanced-baseline-results.json'
    r4 = json.loads((OUT / 'colour299_40.json').read_text())['baseline_black_test']
    result = {'name': name, 'selection': chosen, 'quickdraw': {}, 'external': {},
              'architecture': {'widths': checkpoint['widths'], 'fc': checkpoint['fc'],
                               'size': checkpoint['size'], 'input_channels': 4,
                               'rgb_strength': checkpoint.get('rgb_strength', 1.),
                               'feature_normalize': net.cosine},
              'quality_tier': 'Parked. No quality score or effect tiers fitted.'}
    path = ROOT / 'notes/balanced-training-results.json'
    data = ColourData('test', 'colour299-40')
    for mode, seed in MODES:
        baseline['quickdraw'][mode] = qd(installed_net, data, mode, seed)[0]
        write(baseline_path, baseline)
        metrics, top, _ = qd(net, data, mode, seed)
        result['quickdraw'][mode] = metrics
        np.save(OUT / f'{name}-test-{mode}-top3.npy', top)
        write(path, result)
        print('TEST', mode, metrics['real'], metrics['false_alarms'], flush=True)
    for domain, primary in [('tu', 'line'), ('qd_unrecognized', 'native'), ('clipart', 'preserve'),
                            ('painting', 'preserve'), ('sketch', 'line')]:
        result['external'][domain] = {'primary_mode': primary, 'modes': {}}
        baseline['external'][domain] = {'primary_mode': primary, 'modes': {}}
        modes = [primary] if domain == 'qd_unrecognized' else ['line', 'preserve']
        for mode in modes:
            baseline['external'][domain]['modes'][mode] = external(installed_net, domain, mode)[0]
            write(baseline_path, baseline)
            metrics, top, _, _ = external(net, domain, mode)
            result['external'][domain]['modes'][mode] = metrics
            np.save(OUT / f'{name}-test-{domain}-{mode}-top3.npy', top)
            write(path, result)
            print('TEST', domain, mode, metrics['all'], metrics['real_to_custom'], flush=True)
    result['diverse_synthetic'] = procedural(net)
    result['scripted_probes'] = probes(net)
    result['captured_replays'] = captured_replays(net)
    baseline['diverse_synthetic'] = procedural(installed_net)
    baseline['scripted_probes'] = probes(installed_net)
    baseline['captured_replays'] = captured_replays(installed_net)
    write(baseline_path, baseline)
    with np.load(ROUND / 'generated/test.npz') as source:
        from external_eval import metrics as summarise
        top = []
        baseline_top = []
        with torch.inference_mode():
            for lo in range(0, len(source['y']), 64):
                x = torch.from_numpy(source['x'][lo:lo+64].astype(np.float32) / 255).cuda()
                top.append(net(x.contiguous(memory_format=torch.channels_last)).topk(3, 1).indices.cpu().numpy())
                baseline_top.append(installed_net(x.contiguous(memory_format=torch.channels_last)).topk(3, 1).indices.cpu().numpy())
        result['generated_holdout'] = summarise(source['y'], np.concatenate(top), np.ones(len(source['y']), bool))
        baseline['generated_holdout'] = summarise(source['y'], np.concatenate(baseline_top), np.ones(len(source['y']), bool))
        write(baseline_path, baseline)
    guards = {'validation-selection-passed': selection['validation_eligible']}
    for mode, _ in MODES:
        current, installed = result['quickdraw'][mode], baseline['quickdraw'][mode]
        guards[mode + '-custom-false-alarms'] = current['false_alarms']['rate'] <= installed['false_alarms']['rate']
        guards[mode + '-black-hole'] = current['black_hole_false_alarms']['rate'] <= installed['black_hole_false_alarms']['rate']
        if mode == 'black':
            for metric in ['top1', 'top3']:
                guards['black-real-' + metric] = current['real'][metric] >= installed['real'][metric]
                guards['black-synthetic-' + metric] = current['synthetic'][metric] >= max(installed['synthetic'][metric], r4['synthetic'][metric]) - .002
        if mode in ('filled', 'painted'):
            guards[mode + '-improvement'] = current['real']['top1'] >= installed['real']['top1'] + .01
    for domain, value in result['external'].items():
        mode = value['primary_mode']
        c, b = value['modes'][mode], baseline['external'][domain]['modes'][mode]
        guards[domain + '-custom-false-alarms'] = c['real_to_custom']['rate'] <= b['real_to_custom']['rate']
        guards[domain + '-black-hole'] = c['black_hole_false_alarms']['rate'] <= b['black_hole_false_alarms']['rate']
    guards['dev-gestures'] = result['captured_replays']['correct'] >= baseline['captured_replays']['correct']
    result['installation_guards'] = guards
    result['eligible_for_installation'] = all(guards.values())
    result['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write(path, result)
    write(OUT / f'{name}.json', {
        'name': name, **result['architecture'],
        'int8_top1': result['quickdraw']['black']['all']['top1'],
        'int8_top3': result['quickdraw']['black']['all']['top3'],
        'test_modes': result['quickdraw'], 'installed': False,
        'accepted_black': all(value for key, value in guards.items() if key.startswith('black-')),
        'eligible_for_installation': result['eligible_for_installation'],
        'quality_tier': 'parked',
    })
    print('FINAL', name, 'eligible for installation', result['eligible_for_installation'],
          'failed', [k for k, v in guards.items() if not v], flush=True)


if __name__ == '__main__':
    main()
