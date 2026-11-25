"""Final validation-only classifier trials; leaves all outside tests unopened."""
import json
import subprocess
import sys
import time

from balanced_validation import ROUND, OUT, write, select


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    while not (OUT / 'balanced299_40_headr50_s6000-selection.json').exists():
        time.sleep(10)
    time.sleep(10)
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    new = [f'balanced299_40_affine{suffix}_s{step}' for suffix in ['r25', 'r50'] for step in [2000, 6000]]
    new += [f'balanced299_40_freer25_s{step}' for step in [2000, 6000]]
    protocol['candidates'] = list(dict.fromkeys(protocol['candidates'] + new))
    protocol['classifier_followup'] = 'Validation showed offsets alone lose old custom recall. Test per-custom positive slopes/intercepts and ordinary free-scale custom output weights, preserving all real rows and frozen features. No outside tests opened.'
    write(ROUND / 'protocol.json', protocol)
    for rgb, suffix in [(.25, 'r25'), (.5, 'r50')]:
        name = 'balanced299_40_affine' + suffix
        run('train_balanced_affine.py', '--name', name, '--rgb', str(rgb), '--features', 'balanced299_40_head' + suffix)
        for step in [2000, 6000]:
            run('balanced_validation.py', '--name', name + f'_s{step}')
    run('train_balanced_head.py', '--name', 'balanced299_40_freer25', '--rgb', '.25', '--free-scale', '--reuse-features', 'balanced299_40_headr25')
    for step in [2000, 6000]:
        run('balanced_validation.py', '--name', 'balanced299_40_freer25' + f'_s{step}')
    results = [json.loads((OUT / f'{name}-selection.json').read_text()) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    write(ROUND / 'selection.json', {
        'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    print('FINAL VALIDATION SELECTION', winner['name'], bool(eligible), flush=True)


if __name__ == '__main__':
    main()
