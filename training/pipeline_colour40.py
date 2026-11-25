"""Finish the requested 299-class comparison after CPU preparation / 28px GPU work."""
import json
from pathlib import Path
import subprocess
import sys
import time

if __name__=='__main__':
    here=Path(__file__).resolve().parent;root=here.parent;out=here/'out'
    def run(log,script,*args):
        with (out/log).open('w',encoding='utf-8',newline='\n') as f:
            subprocess.run([sys.executable,'-u',str(here/script),*args],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    while not (root/'data/colour299-40/corpus.json').exists():time.sleep(5)
    run('prepare-fill40.log','prepare_fill.py','--folder','colour299-40','--threads','6')
    while True:
        status=json.loads((out/'colour-status.json').read_text())
        if status['phase']=='complete':break
        if status['phase']=='failed':raise RuntimeError('28px pipeline failed; do not start competing GPU training')
        time.sleep(5)
    run('colour40.log','train_colour.py','--name','colour299_40','--size','40','--epochs','3','--shape-lr','0.0001','--colour-seed','20461109','--bs','1024')
    run('colour40.export.log','export.py','--name','colour299_40','--samples','299','--out-dir',str(root/'luau/models/colour299_40'))
    print('40px comparison ready; installation remains a separate acceptance decision.',flush=True)
