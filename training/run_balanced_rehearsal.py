"""Rehearse from the entire old synthetic training range, with fresh black ink."""
import json
import subprocess
import sys
import time
from balanced_validation import ROUND, OUT, write


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], check=True)


def main():
    while not (OUT / 'balanced299_40_branch768_margin_s6000-selection.json').exists():
        time.sleep(10)
    time.sleep(10)
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    names = ['balanced299_40_branch768_rehearsal_s2000', 'balanced299_40_branch768_rehearsal_s10000']
    protocol['candidates'] = list(dict.fromkeys(protocol['candidates'] + names))
    protocol['full_old_training_rehearsal'] = 'Previous feature bank sampled 1,000 original synthetic training drawings per class. Now sample 64 fresh black anchors per minibatch from ALL original synthetic TRAIN IDs, protecting the full range of old generator variants. No validation/test drawing enters gradient updates.'
    write(ROUND / 'protocol.json', protocol)
    run('train_balanced_branch.py', '--name', 'balanced299_40_branch768_rehearsal', '--steps', '10000',
        '--resume', 'balanced299_40_branch768_margin_s6000.pt', '--reuse-features', 'balanced299_40_branch768',
        '--hard-bank', 'branch-hard-negatives.npz', '--retention', '5', '--old-margin-weight', '20', '--fresh-old-black')
    for name in names:
        run('balanced_validation.py', '--name', name)
    run('freeze_balanced_selection.py')
    print('REHEARSAL ROUND COMPLETE', flush=True)


if __name__ == '__main__':
    main()
