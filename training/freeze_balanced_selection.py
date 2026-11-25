"""Final uniform calibration using the author’s exact acceptance thresholds."""
import json
import time

from balanced_validation import ROUND, write, select


def main():
    if (ROUND / 'test-evaluation-started.json').exists():
        raise RuntimeError('This round is frozen. Start a new round and model prefix.')
    protocol = json.loads((ROUND / 'protocol.json').read_text())
    protocol['calibration_revision'] = 5
    protocol['calibration'] = 'Per-output constrained offsets with calibration FA counts no greater than installed and custom recall within .2pp. Conservative union bound and exact calibration recall checks. Shared other-28/BH offsets as fallback. Independent guard halves apply unchanged exact user acceptance bars.'
    protocol['calibration_revision_reason'] = 'The extra 10-20% FA safety buffer and .1pp calibration recall buffer unnecessarily traded away old custom recall. Use the requested no-worse FA / .2pp retention bars, then verify the independent guard half. No outside or developer test inputs have been read.'
    write(ROUND / 'protocol.json', protocol)
    results = [select(name) for name in protocol['candidates']]
    eligible = [result for result in results if result['eligible']]
    winner = max(eligible, key=lambda result: result['rank']) if eligible else min(
        results, key=lambda result: (sum(not value for value in result['guards'].values()), -result['rank']))
    write(ROUND / 'selection.json', {
        'selected': winner['name'], 'validation_eligible': bool(eligible), 'candidates': results,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    print('FROZEN SELECTION', winner['name'], bool(eligible), flush=True)


if __name__ == '__main__':
    main()
