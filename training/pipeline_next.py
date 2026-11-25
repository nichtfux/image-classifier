"""Reproduce the separate colour/fill/GACL candidate, entirely offline.

Requires the pinned recognized-QD colour299-40 corpus and colour299_40 fallback.
Nothing here connects to Studio or installs a model.
"""
import argparse,subprocess,sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',default='2.6');ap.add_argument('--epochs',default='3')
    ap.add_argument('--reuse-synth',action='store_true');a=ap.parse_args()
    here=Path(__file__).resolve().parent
    assert (here.parent/'data/colour299-40/corpus.json').exists()
    assert (here/'out/colour299_40.pt').exists()
    def run(script,*args):subprocess.run([sys.executable,'-u',script,*args],cwd=here,check=True)
    if not a.reuse_synth:run('synth_diverse.py')
    run('train_next.py','--hours',a.hours,'--epochs',a.epochs)
    run('select_next.py')
    checkpoint=(here/'out/gacl299_40_wide-selected.txt').read_text().strip()
    run('export_candidate.py','--checkpoint',checkpoint)
    run('evaluate_next.py','--baseline-only');run('evaluate_next.py')
    run('export_candidate.py','--checkpoint',checkpoint)
    run('offline_native.py','--candidate',str(here.parent/'luau/models/gacl299_40_wide'),'--tag','gacl299_40_wide')
    run('report_next.py')


if __name__=='__main__':main()
