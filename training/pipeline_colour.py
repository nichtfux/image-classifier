"""Colour warm-up/refinement with retained r4 fallback; never writes Studio."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--skip-prepare',action='store_true')
    ap.add_argument('--wait-for-prepare',action='store_true');ap.add_argument('--skip-rgb',action='store_true')
    ap.add_argument('--skip-refine',action='store_true');ap.add_argument('--compare40',action='store_true');a=ap.parse_args()
    here=Path(__file__).resolve().parent;root=here.parent;out=here/'out';completed=[]
    (out/'colour-status.json').write_text(json.dumps({'phase':'starting','completed':[]}),encoding='utf-8',newline='\n')
    def run(name,script,*args):
        command=[sys.executable,'-u',str(here/script),*args]
        with (out/name).open('w',encoding='utf-8',newline='\n') as log:
            p=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            (out/'colour-status.json').write_text(json.dumps({'phase':name,'pid':p.pid,'command':command,'completed':completed},indent=1),encoding='utf-8',newline='\n')
            code=p.wait()
        if code:
            (out/'colour-status.json').write_text(json.dumps({'phase':'failed','failed_phase':name,'exit_code':code,'completed':completed}),encoding='utf-8',newline='\n')
            raise subprocess.CalledProcessError(code,command)
        completed.append(name)
    if a.wait_for_prepare:
        deadline=time.time()+3600
        while not (root/'data/colour299-28/corpus.json').exists():
            if time.time()>deadline:raise TimeoutError('Colour metadata not ready')
            time.sleep(2)
    elif not a.skip_prepare:run('prepare-colour.log','prepare_colour.py')
    forty_prep=None;forty_worker=None;forty_logs=[]
    if a.compare40:
        if not (root/'data/colour299-40/corpus.json').exists():
            f=(out/'prepare-colour40.log').open('w',encoding='utf-8',newline='\n');forty_logs.append(f)
            forty_prep=subprocess.Popen([sys.executable,'-u',str(here/'prepare_colour.py'),'--out','colour299-40','--size','40','--workers','8'],cwd=root,stdout=f,stderr=subprocess.STDOUT)
        f=(out/'colour40-pipeline.log').open('w',encoding='utf-8',newline='\n');forty_logs.append(f)
        forty_worker=subprocess.Popen([sys.executable,'-u',str(here/'pipeline_colour40.py')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    run('prepare-fill28.log','prepare_fill.py','--folder','colour299-28')
    if not a.skip_rgb:run('colour-rgb.log','train_colour.py','--name','colour299_rgb','--rgb-only','--epochs','1','--bs','512')
    if not a.skip_refine:run('colour-refine.log','train_colour.py','--name','colour299_full','--epochs','1',
            '--resume',str(out/'colour299_rgb.pt'),'--colour-seed','20361109','--bs','1024','--shape-lr','0.00002')
    candidates=[]
    for name in ('colour299_rgb','colour299_full'):
        path=out/(name+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            if result['accepted']:
                score=max((m['policy']['all']['top1']+m['filled']['all']['top1'])/2 for m in result['history'] if m['guard'])
                candidates.append((score,name))
    assert candidates,'No candidate passed black retention; keep r4'
    candidates.sort(reverse=True);selected=candidates[0][1]
    run(selected+'.export.log','export.py','--name',selected,'--samples','299','--out-dir',str(root/'luau/models'/selected))
    (out/'colour-selection.json').write_text(json.dumps({'selected':selected,'validation_ranking':candidates,
        'fallback':'wide299_28_r4','default_export_unchanged':True},indent=1),encoding='utf-8',newline='\n')
    (out/'colour-status.json').write_text(json.dumps({'phase':'complete','selected':selected,'completed':completed},indent=1),encoding='utf-8',newline='\n')
    if forty_prep:assert forty_prep.wait()==0,'40px data preparation failed'
    if forty_worker:assert forty_worker.wait()==0,'40px comparison failed'
    for f in forty_logs:f.close()
