"""Background sequence. Stops before final test evaluation or installation."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'out'
ROUND = ROOT / 'data/generated/rounds/balanced-v4-20261110'


def run(*arguments):
    subprocess.run([sys.executable, '-u', *arguments], cwd=HERE, check=True)


def main():
    while not (ROUND / 'qd-game/manifest.json').exists():
        time.sleep(10)
    protocol = {
        'source': 'gacl299_40_continued', 'conversion': 'generated-strokes-v4',
        'snapshot': str(ROUND), 'seed': 2026111102,
        'training': 'QD recognized train + own procedural train + v4 generated train only',
        'generated_batch_balance': {'real': 6, 'custom': 2, 'qd_v4_real': 8},
        'candidates': ['balanced299_40_control', 'balanced299_40_n025_s1500',
                       'balanced299_40_n025_s4500', 'balanced299_40_n075_s1500',
                       'balanced299_40_n075_s4500'],
        'selection': 'All pinned QD validation, own generated validation and converted QD validation only',
        'calibration': 'First 500 QD validation rows per class plus corresponding own validation half. Minimum nonnegative offset per custom output, each targeting half its installed false-positive count per destination and calibration cohort.',
        'guards': 'Second 500 rows per class and separate own validation half. Real black top1/top3 >= installed. Old custom black top1/top3 within .2pp. All validation modes custom/BH FA <= installed. Filled/painted real top1 >= installed+1pp. Policy/random real top1 within .2pp.',
        'rank': 'Mean filled and painted real validation top1 among eligible candidates',
        'outside': 'Evaluate frozen best eligible candidate once. Never used for selection or calibration.',
        'no_eligible_candidate': 'Freeze the candidate with fewest failed validation guards, breaking ties by style rank, for a diagnostic test-only comparison. It cannot be installed.',
        'install': 'Only if test confirms styles improve, black QD does not regress and real-to-custom/BH FA no worse than installed. Otherwise retain installed.',
        'quality': 'Parked until the author ratings',
        'source_sha256': hashlib.sha256((OUT / 'gacl299_40_continued.pt').read_bytes()).hexdigest(),
        'snapshot_sha256': hashlib.sha256((ROUND / 'generated/manifest.json').read_bytes()).hexdigest(),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (ROUND / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8', newline='\n')
    shutil.copy2(OUT / 'gacl299_40_continued.pt', OUT / 'balanced299_40_control.pt')
    run('balanced_validation.py', '--name', 'colour299_40')
    run('balanced_validation.py', '--name', 'balanced299_40_control')
    for strength, suffix in [(.25, 'n025'), (.75, 'n075')]:
        name = 'balanced299_40_' + suffix
        run('train_balanced.py', '--name', name, '--negative', str(strength))
        for steps in [1500, 4500]:
            run('balanced_validation.py', '--name', name + f'_s{steps}')
    results = [json.loads((OUT / f'{name}-selection.json').read_text()) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    frozen = {'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
              'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (ROUND / 'selection.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8', newline='\n')
    print('SELECTION COMPLETE', frozen['selected'], flush=True)


if __name__ == '__main__':
    main()
