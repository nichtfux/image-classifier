"""Human-readable report and standalone comparison figure for the v4 round."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from balanced_validation import ROUND, OUT

ROOT = OUT.parent.parent
NOTES = ROOT / 'notes'


def percentage(value):
    return f'{100 * value:.2f}%'


def rate(value):
    return f'{100 * value:.4f}%'


def score(group):
    if group['top1'] is None:
        return '—'
    return f"{percentage(group['top1'])} / {percentage(group['top3'])}"


def main():
    result = json.loads((NOTES / 'balanced-training-results.json').read_text())
    baseline = json.loads((NOTES / 'balanced-baseline-results.json').read_text())
    selection = json.loads((ROUND / 'selection.json').read_text())
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    status = json.loads((ROOT / 'data/generated/generation-status.json').read_text())
    generated = json.loads((ROUND / 'generated/manifest.json').read_text())
    head = json.loads((ROUND / 'head-generated/manifest.json').read_text())
    newest = json.loads((ROUND / 'raw-generated/manifest.json').read_text())
    negatives = json.loads((ROUND / 'qd-game/manifest.json').read_text())
    installed = result.get('studio', {}).get('installed', False)
    lines = [
        '# Balanced generated-data round, 2026-11-10', '',
        f"Selected candidate `{result['name']}`. " + ('Installed privately in NN test.' if installed else 'Installation rejected by the acceptance guards.'), '',
        'The author approved continuing without his drawings. Every training sample came from recognized Quick, Draw! training data or our own code/FLUX generation. TU Berlin, DomainNet, QD unrecognised and the twelve retained developer gestures stayed outside training, calibration and selection.', '',
        '## What changed', '',
        '- Initial full-network trials used two generated custom slots against six generated real slots and eight game-converted real negatives. The latter cover all 270 real classes.',
        f"- The first immutable v4 bank contains {len(generated['source_rows']):,} kept examples. The follow-up bank contains {len(head['source_rows']):,}. Training, validation and test groups are separated by complete class/prompt hashes. Newly generated validation/test rows were excluded from training.",
        f"- The final capacity trials use a third immutable v4 bank with {len(newest['source_rows']):,} kept examples. The permitted validation cohort remains fixed for paired model comparisons.",
        f"- The conversion-matched real bank contains {negatives['counts']['train']:,} training and {negatives['counts']['validation']:,} validation examples. Illegal/empty conversions were skipped. These are deliberately difficult artificial variants and can lose semantic detail.",
        '- Full-network fine-tunes failed permitted validation. A second approach froze the feature extractor and all 270 real output rows, then learned the 29 custom rows and their biases with real-negative and custom-positive margins.',
        '- The final source is the earlier wide control without FLUX images, `gacl299_40_continued`. Another 384 custom-only hidden units bring its dense layer to 768. Its original real-class path stays frozen. The selected 512-item batch uses 96 QD real, 192 mined QD real, 80 game-converted QD real, 40 generated real, 64 fresh old synthetic black, 24 broader procedural and 16 generated custom examples.',
        '- Fresh old synthetic anchors are sampled from the entire original synthetic training range. Earlier trials rehearsed only 1,000 examples per class. Correct original custom margins are protected, and the 34,854 mined negatives come exclusively from QD training rows 5,000,000 through 5,999,999.',
        '- RGB strengths 0, 0.25 and 0.5 were checked against the full-strength control. Zero RGB was a diagnostic control and could not be installed. The retained colour variants still take four input planes.',
        f"- The selected network uses convolution widths {result['architecture']['widths']}, {result['architecture']['fc']} hidden units, RGB strength {result['architecture']['rgb_strength']}, and feature normalization {result['architecture']['feature_normalize']}.",
        '- Custom and separate black-hole bias offsets were fitted on calibration halves, then checked against separate validation halves. The calibration revisions and their validation-only reasons are recorded in the protocol. No outside score influenced those revisions.',
        '- The selected model uses zero additional post-training output offsets. Its learned network produced the lower QD false-alarm rates.',
        '- The int8 format and preview geometry remain 40px, four padded planes. The exporter now preserves nonzero output biases. Independent float64 checks cover that arithmetic.',
        '- Quality/detail tiers remain parked. No new quality threshold or game effect was built.', '',
        '## Selection flaw found in the final audit', '',
        'The original selection guards protected black custom drawings, while style improvement was checked on the 270 real classes. That let a large filled-custom regression pass validation. Its lower filled-custom validation score was already present in the cached report. I should have included that cohort in selection.', '',
        'Future selection now also guards custom top-one/top-three accuracy for filled and brush styles. Selection is locked once final tests open. The frozen candidate was not retuned or reselected after seeing the outside sets or developer gestures.', '',
        'The filled-red-heart probe changes from heart to diamond. This reinforces the rejection. [Forced-fill diagnosis on code-generated training drawings](custom-augmentation-diagnosis.png) shows a worst case where blanket black filling erases spiral curls or hat details. These examples were selected by closed-region area, not sampled to estimate frequency.', '',
        '[Identical probe inputs with the installed and candidate answers](balanced-regression-examples.png) show the filled heart regression alongside the repaired top-fuse bomb.', '',
        '## Held-out Quick, Draw! and original synthetic test', '',
        'First guess / top three. Real and synthetic groups are reported separately. Synthetic results are an upper bound.', '',
        '| Input style | Installed real | Candidate real | Installed synthetic | Candidate synthetic |',
        '|---|---:|---:|---:|---:|',
    ]
    modes = [('black', 'Black outline'), ('policy', 'Colour table'), ('random', 'Random colours'),
             ('filled', 'Filled'), ('painted', 'Paint / brush widths')]
    for key, label in modes:
        b, c = baseline['quickdraw'][key], result['quickdraw'][key]
        lines.append(f"| {label} | {score(b['real'])} | {score(c['real'])} | {score(b['synthetic'])} | {score(c['synthetic'])} |")
    lines += ['', 'Colour-table evaluation measures the authored colour rules. Filled and brush evaluations measure artificial augmentation. None establishes real-player accuracy.', '',
              '## Real drawings routed into a custom class', '',
              '| Style | Installed custom FA | Candidate custom FA | Installed black hole | Candidate black hole |',
              '|---|---:|---:|---:|---:|']
    for key, label in modes:
        b, c = baseline['quickdraw'][key], result['quickdraw'][key]
        lines.append(f"| {label} | {rate(b['false_alarms']['rate'])} | {rate(c['false_alarms']['rate'])} | {rate(b['black_hole_false_alarms']['rate'])} | {rate(c['black_hole_false_alarms']['rate'])} |")
    lines += ['', '## Outside tests', '',
              '| Set | Installed first / top three | Candidate first / top three | Installed custom FA | Candidate custom FA | Installed BH FA | Candidate BH FA |',
              '|---|---:|---:|---:|---:|---:|---:|']
    domains = [('tu', 'TU Berlin'), ('qd_unrecognized', 'QD unrecognised'), ('clipart', 'DomainNet clipart'),
               ('painting', 'DomainNet painting'), ('sketch', 'DomainNet sketch')]
    for key, label in domains:
        mode = result['external'][key]['primary_mode']
        b, c = baseline['external'][key]['modes'][mode], result['external'][key]['modes'][mode]
        lines.append(f"| {label} | {score(b['all'])} | {score(c['all'])} | {rate(b['real_to_custom']['rate'])} | {rate(c['real_to_custom']['rate'])} | {rate(b['black_hole_false_alarms']['rate'])} | {rate(c['black_hole_false_alarms']['rate'])} |")
    lines += ['', '### Custom-class generalisation', '',
              '| Custom cohort | Installed first / top three | Candidate first / top three |',
              '|---|---:|---:|']
    for key, label in domains:
        mode = result['external'][key]['primary_mode']
        b, c = baseline['external'][key]['modes'][mode]['custom'], result['external'][key]['modes'][mode]['custom']
        if c['n']:
            lines.append(f"| {label} mapped custom classes, n={c['n']} | {score(b)} | {score(c)} |")
    lines.append(f"| Own broader procedural test | {score(baseline['diverse_synthetic']['all'])} | {score(result['diverse_synthetic']['all'])} |")
    lines.append(f"| Own v4 generated test, custom | {score(baseline['generated_holdout']['custom'])} | {score(result['generated_holdout']['custom'])} |")
    lines += ['', 'Both own-generated test cohorts are upper bounds. Their style and teacher filtering are correlated with training. They are not substitutes for real player drawings.', '',
              'Worst custom destinations on the real QD test', '']
    for key, label in modes:
        worst = result['quickdraw'][key]['false_alarms']['worst'][:5]
        lines.append('- ' + label + ' → ' + ', '.join(f"{row['label']} {row['count']}" for row in worst))
    lines += ['', 'Mappings, conversion modes and test IDs match the previous evaluation. Raster adapters cannot recover stroke ownership, widths or layering. Painting/clipart foreground extraction and palette conversion introduce strong bias. DomainNet licensing remains unconfirmed and it remains offline test-only. [Adapter and licensing caveats](external-evaluation.md).', '',
              '## Twelve retained developer gestures', '',
              f"Installed {baseline['captured_replays']['correct']}/12. Candidate {result['captured_replays']['correct']}/12. These are self-authored MCP captures, with no real-player accuracy claim.", '',
              '| Gesture | Installed first | Candidate top three |', '|---|---|---|']
    for b, c in zip(baseline['captured_replays']['records'], result['captured_replays']['records']):
        lines.append(f"| {c['label']}, {c['paint']}, {c['nominal_scale']} | {b['top3'][0]['label']} | " + ', '.join(r['label'] for r in c['top3']) + ' |')
    lines += ['', '### Minimal top-fuse bomb', '']
    for row in result['scripted_probes']:
        if 'bomb' in row['name'].lower():
            lines.append('- ' + row['name'] + ' → ' + ', '.join(f"{r['label']} {percentage(r['prob'])}" for r in row['top3']))
    lines += ['', '## Validation selection and acceptance', '',
              '| Candidate | Failed validation guards | Mean filled / painted real accuracy |', '|---|---|---:|']
    for row in selection['candidates']:
        failed = [k for k, value in row['guards'].items() if not value]
        lines.append(f"| {row['name']} | {', '.join(failed) or 'Passed'} | {percentage(row['rank'])} |")
    lines += ['', 'Final installation guards that failed', '']
    failures = [key for key, value in result['installation_guards'].items() if not value]
    lines += ['- ' + value for value in failures] or ['- None']
    lines += ['', f"The selected model was frozen at {selection['frozen_utc']} before its final test evaluation. [Protocol](../data/generated/rounds/balanced-v4-20261110/protocol.json), [complete results](balanced-training-results.json).", '',
              'The no-worse custom/black-hole false-alarm rule was applied separately to every outside set. DomainNet sketch fails that rule. Aggregate improvements do not remove this recorded per-set failure.', '',
              '## Runtime checks', '']
    if 'runtime' in result:
        runtime = result['runtime']
        rows = runtime['candidate']
        lines += ['| Standalone Luau 0.746 | Median forward | 95th percentile | Maximum forward |',
                  '|---|---:|---:|---:|']
        for mode in ('native', 'interpreted'):
            values = rows[mode]['FORWARD']
            lines.append(f'| {mode} | {values[0]:.2f} ms | {values[1]:.2f} ms | {values[2]:.2f} ms |')
        peak = max(rows['native'][key][1] for key in ('BURST8', 'RECOGNIZE8', 'COMPLEX_RECOGNIZE'))
        active = rows['native']['BURST8'][0]
        lines += ['', f'Native peak cooperative slice in these fixtures was {peak:.2f} ms. Eight interleaved forwards used {active:.2f} ms active work, equivalent to {active / 120:.2f} seconds at an ideal 2ms/60Hz budget.', '',
                  f"Calibration against the prior 6.87ms Studio baseline estimates {runtime['calibrated_server_forward_ms']:.2f} ms per forward. This is an extrapolation from standalone CLI timing under concurrent generation. It is not a fresh server measurement. No Studio calls were made because the candidate failed acceptance.", '',
                  'The independent float64 reference matches in native and interpreted execution. Sliced and synchronous results, interleaved jobs, raster inputs and all top guesses match. The exported int8 arrays are the frozen arrays used for final evaluation, with no CPU requantization afterward.', '']
    remaining = sum(max(0, (100 if index < 270 else 5000) - row['kept']) for index, row in enumerate(status['classes']))
    days = remaining / max(status['kept_per_second'], 1e-9) / 86400
    lines += [
              '## Background generation', '',
              f"Latest status has {status['attempted']:,} attempted and {status['kept']:,} kept images. Worker {status['pid']} is {status['status']}. It hit a transient Windows status-file replacement lock and resumed from the intact manifest after adding bounded retries.", '',
              'Source weights, pinned revisions, licences and hashes remain recorded in data/generated/SOURCES.md. Every generated image is marked generated. The worker targets 5,000 kept custom images per class and 100 per real class, with attempt/zero-keeper caps.', '',
              f"The current session averages {status['images_per_second']:.2f} attempted and {status['kept_per_second']:.2f} kept images per second. A simple projection of the remaining target is {days:.1f} continuous days at that loaded-PC rate. The earlier idle benchmark was faster. Low-keeper classes can hit their attempt cap before the target, so this is a throughput projection rather than a completion promise.", '',
              '## Not checked', '',
              'The author has no exported held-out drawings. Real filled/coloured player accuracy and Rough/Good/Detailed quality tiers remain unmeasured. Fresh Studio timing and parity are recorded only if installation passes its gates.', '']
    (NOTES / 'balanced-training-round.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    colors = ['#66788f', '#17836d']
    labels = [label for _, label in modes]
    positions = np.arange(len(modes))
    for j, (model, title) in enumerate([(baseline, 'Installed'), (result, 'Candidate')]):
        values = [model['quickdraw'][key]['real']['top1'] * 100 for key, _ in modes]
        bars = axes[0, 0].bar(positions + (j-.5)*.36, values, .36, label=title, color=colors[j])
        axes[0, 0].bar_label(bars, fmt='%.1f', fontsize=10)
    axes[0, 0].set_xticks(positions, ['Black', 'Colour', 'Random', 'Filled', 'Brush'])
    axes[0, 0].set_ylim(0, 90)
    axes[0, 0].set_title('First-guess accuracy on real QD test drawings (%)', loc='left', weight='bold')
    axes[0, 0].legend(frameon=False)
    for j, (model, title) in enumerate([(baseline, 'Installed'), (result, 'Candidate')]):
        values = [model['quickdraw'][key]['synthetic']['top1'] * 100 for key, _ in modes]
        bar_colors = [colors[j]] * len(modes)
        if j == 1:
            bar_colors[3] = '#bd3e4f'
        bars = axes[0, 1].bar(positions + (j-.5)*.36, values, .36, label=title, color=bar_colors)
        axes[0, 1].bar_label(bars, fmt='%.1f', fontsize=10)
    axes[0, 1].set_xticks(positions, ['Black', 'Colour', 'Random', 'Filled', 'Brush'])
    axes[0, 1].set_ylim(0, 110)
    axes[0, 1].set_title('Original custom drawings. Filled regression in red (%)', loc='left', weight='bold')
    for j, (model, title) in enumerate([(baseline, 'Installed'), (result, 'Candidate')]):
        values = [model['external'][key]['modes'][model['external'][key]['primary_mode']]['all']['top1'] * 100 for key, _ in domains]
        bars = axes[1, 0].bar(positions + (j-.5)*.36, values, .36, color=colors[j])
        axes[1, 0].bar_label(bars, fmt='%.1f', fontsize=10)
    axes[1, 0].set_xticks(positions, ['TU Berlin', 'Failed QD', 'Clipart', 'Painting', 'Sketch'])
    axes[1, 0].set_ylim(0, 75)
    axes[1, 0].set_title('Outside test accuracy, adapter caveats apply (%)', loc='left', weight='bold')
    axes[1, 1].axis('off')
    bq, cq = baseline['quickdraw']['black'], result['quickdraw']['black']
    state = 'Installed in NN test' if installed else 'Acceptance gates blocked installation'
    filled_b = baseline['quickdraw']['filled']['synthetic']['top1']
    filled_c = result['quickdraw']['filled']['synthetic']['top1']
    sketch_b = baseline['external']['sketch']['modes']['line']['black_hole_false_alarms']['rate']
    sketch_c = result['external']['sketch']['modes']['line']['black_hole_false_alarms']['rate']
    text = (f"{state}\n\n"
            f"Filled custom accuracy\n{percentage(filled_b)} → {percentage(filled_c)}\n"
            'Filled red heart changes to diamond.\n\n'
            f"Sketch black-hole false alarms\n{percentage(sketch_b)} → {percentage(sketch_c)}\n\n"
            f"Real QD custom false alarms\n{percentage(bq['false_alarms']['rate'])} → {percentage(cq['false_alarms']['rate'])}\n\n"
            f"Retained developer gestures\n{baseline['captured_replays']['correct']}/12 → {result['captured_replays']['correct']}/12\n\n"
            'Filled custom drawings were missing\nfrom the original selection guards.')
    axes[1, 1].text(.03, .98, text, va='top', fontsize=14, linespacing=1.3)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Balanced conversion-v4 training round', fontsize=24, weight='bold')
    fig.savefig(NOTES / 'balanced-training-results.png', dpi=150)
    plt.close(fig)
    print('Wrote report and figure', flush=True)


if __name__ == '__main__':
    main()
