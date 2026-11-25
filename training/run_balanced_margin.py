"""Restore old correct custom margins while retaining mined-negative training."""
import json
import subprocess
import sys
from balanced_validation import ROUND, OUT, write


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    names = ['balanced299_40_branch768_margin_s2000', 'balanced299_40_branch768_margin_s6000']
    protocol['candidates'] = list(dict.fromkeys(protocol['candidates'] + names))
    protocol['old_margin_rehearsal'] = 'On old synthetic TRAIN anchors that the frozen original classifier gets right, preserve its true-class/rival margin up to 2 logits. Weight 20, mined real negatives retained, output L2 retention 5. Targets the sole remaining validation failure, old custom recall. No test data read.'
    write(ROUND / 'protocol.json', protocol)
    run('train_balanced_branch.py', '--name', 'balanced299_40_branch768_margin',
        '--resume', 'balanced299_40_branch768_hard_s6000.pt', '--reuse-features', 'balanced299_40_branch768',
        '--hard-bank', 'branch-hard-negatives.npz', '--retention', '5', '--old-margin-weight', '20')
    for name in names:
        run('balanced_validation.py', '--name', name)
    import time
    results = [json.loads((OUT / f'{name}-selection.json').read_text()) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    write(ROUND / 'selection.json', {
        'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    print('MARGIN FINAL SELECTION', winner['name'], bool(eligible), flush=True)


if __name__ == '__main__':
    main()
