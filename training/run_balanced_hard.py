"""Train once on mined training-only confusions, then freeze final selection."""
import json
import subprocess
import sys
import time

from balanced_validation import ROUND, OUT, write, select


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    while not (OUT / 'branch-hard-negatives.json').exists():
        time.sleep(10)
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    names = ['balanced299_40_branch768_hard_s2000', 'balanced299_40_branch768_hard_s6000']
    protocol['candidates'] = list(dict.fromkeys(protocol['candidates'] + names))
    protocol['hard_negative_pass'] = 'Mine recognized QD TRAIN rows 5,000,000..6,000,000, with independently seeded black/fill/brush augmentation. Keep near-custom cases, at most 4,000 per confused custom destination. Replace 192/512 training slots with these real negatives and strengthen old synthetic residual retention to 5. Validation/outside/dev gestures never supply mining inputs.'
    write(ROUND / 'protocol.json', protocol)
    run('train_balanced_branch.py', '--name', 'balanced299_40_branch768_hard',
        '--resume', 'balanced299_40_branch768_s6000.pt', '--reuse-features', 'balanced299_40_branch768',
        '--hard-bank', 'branch-hard-negatives.npz', '--retention', '5')
    for name in names:
        run('balanced_validation.py', '--name', name)
    results = [json.loads((OUT / f'{name}-selection.json').read_text()) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    write(ROUND / 'selection.json', {
        'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    print('HARD PASS FINAL SELECTION', winner['name'], bool(eligible), flush=True)


if __name__ == '__main__':
    main()
