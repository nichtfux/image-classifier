"""Final capacity trial, preserving existing real-class features and outputs."""
import json
import subprocess
import sys
import time

from balanced_validation import ROUND, OUT, write, select


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    while not (OUT / 'balanced299_40_rawr50_s6000-selection.json').exists():
        time.sleep(10)
    time.sleep(10)
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    names = ['balanced299_40_branch768_s2000', 'balanced299_40_branch768_s6000']
    protocol['candidates'] = list(dict.fromkeys(protocol['candidates'] + names))
    protocol['residual_capacity'] = 'Add 384 custom-only hidden units, total fc768. Preserve old 384 hidden units, all real output rows, and original custom feature weights. New custom residuals are penalized on old synthetic anchors. Same generic three-conv/two-dense export graph. Expected native work remains far under 100ms; measure after selection.'
    write(ROUND / 'protocol.json', protocol)
    run('train_balanced_branch.py')
    for name in names:
        run('balanced_validation.py', '--name', name)
    results = [select(name) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    write(ROUND / 'selection.json', {
        'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    print('BRANCH FINAL SELECTION', winner['name'], bool(eligible), flush=True)


if __name__ == '__main__':
    main()
