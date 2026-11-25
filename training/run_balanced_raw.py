"""Check ordinary unnormalized hidden features after quality-tier parking."""
import json
import subprocess
import sys
import time

from balanced_validation import ROUND, OUT, write, select
from balanced_data import freeze_v4


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    while not (OUT / 'balanced299_40_freer25_s6000-selection.json').exists():
        time.sleep(10)
    time.sleep(10)
    if not (ROUND / 'raw-generated/manifest.json').exists():
        freeze_v4(ROUND / 'raw-generated')
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    new = [f'balanced299_40_raw{suffix}_s{step}' for suffix in ['r25', 'r50'] for step in [2000, 6000]]
    protocol['candidates'] = list(dict.fromkeys(protocol['candidates'] + new))
    protocol['raw_feature_control'] = 'Compare ordinary unnormalized features with the GACL normalized head. Real-class weight directions and all feature layers remain frozen. Hidden magnitude is available to classification, never mapped to quality tiers.'
    protocol['raw_training_snapshot'] = str(ROUND / 'raw-generated')
    write(ROUND / 'protocol.json', protocol)
    for rgb, suffix in [(.25, 'r25'), (.5, 'r50')]:
        name = 'balanced299_40_raw' + suffix
        run('train_balanced_head.py', '--name', name, '--rgb', str(rgb), '--free-scale', '--raw-features', '--generated-folder', 'raw-generated')
        for step in [2000, 6000]:
            run('balanced_validation.py', '--name', name + f'_s{step}')
    # All candidates receive the same final calibration method. Only already
    # cached permitted validation logits are read by this last comparison.
    results = [select(name) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    write(ROUND / 'selection.json', {
        'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    print('FROZEN FINAL SELECTION', winner['name'], bool(eligible), flush=True)


if __name__ == '__main__':
    main()
