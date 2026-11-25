"""Validation-guided follow-up after full fine-tunes failed retention guards."""
import json
import subprocess
import sys
import time

from balanced_data import freeze_v4
from balanced_validation import ROUND, OUT, write, select


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    while not (ROUND / 'selection.json').exists():
        time.sleep(10)
    if not (ROUND / 'head-generated/manifest.json').exists():
        freeze_v4(ROUND / 'head-generated')
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    new_names = [f'balanced299_40_head{suffix}_s{step}' for suffix in ['r0', 'r25', 'r50'] for step in [2000, 6000]]
    protocol['candidates'] = list(dict.fromkeys(protocol['candidates'] + new_names))
    protocol['calibration_revision'] = 3
    protocol['calibration'] = 'Minimum shared other-28 offset plus separate BH offset, targeting 80% installed false positives on calibration halves. Independent guard halves remain unchanged.'
    protocol['calibration_revision_reason'] = 'Validation-only full fine-tunes forgot real classes. Per-destination zero-error constraints were overly conservative for rare classes. New trials freeze features and all 270 real outputs, learn only the custom outputs, and test RGB strength 0/.25/.5.'
    protocol['head_training_snapshot'] = str(ROUND / 'head-generated')
    protocol['zero_rgb'] = 'Diagnostic control only, ineligible for installation. Coloured model functionality must be retained.'
    write(ROUND / 'protocol.json', protocol)
    for name in protocol['candidates'][:5]:
        select(name)
    for rgb, suffix in [(0., 'r0'), (.25, 'r25'), (.5, 'r50')]:
        name = 'balanced299_40_head' + suffix
        run('train_balanced_head.py', '--name', name, '--rgb', str(rgb))
        for step in [2000, 6000]:
            run('balanced_validation.py', '--name', name + f'_s{step}')
    results = [json.loads((OUT / f'{name}-selection.json').read_text()) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    write(ROUND / 'selection.json', {
        'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    print('HEAD SELECTION COMPLETE', winner['name'], bool(eligible), flush=True)


if __name__ == '__main__':
    main()
