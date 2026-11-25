"""Finish the loss-reduction correction without touching the frozen pilot bank."""
import json,subprocess,sys
from pathlib import Path
import psutil

HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'data/generated';OUT=HERE/'out'


def run(script,*args,logname):
    print('RUN',script,*args,flush=True)
    with (OUT/logname).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-u',script,*args],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)


if __name__=='__main__':
    info=json.loads((DATA/'overnight-process.json').read_text())
    try:psutil.Process(info['pid']).wait()
    except psutil.NoSuchProcess:pass
    run('train_generated_ablation.py','--only-distill',logname='generated-scaled-kd-train.log')
    checkpoint=str(OUT/'gacl299_40_genkd.pt')
    run('export_candidate.py','--checkpoint',checkpoint,'--name','gacl299_40_genkd',logname='generated-scaled-kd-export.log')
    run('evaluate_next.py','--name','gacl299_40_genkd',logname='generated-scaled-kd-evaluation.log')
    run('export_candidate.py','--checkpoint',checkpoint,'--name','gacl299_40_genkd',logname='generated-scaled-kd-final-export.log')
    run('evaluate_generated_holdout.py',logname='generated-scaled-kd-holdout.log')
    run('offline_native.py','--candidate',str(HERE.parent/'luau/models/gacl299_40_genkd'),'--tag','gacl299_40_genkd',logname='generated-scaled-kd-native.log')
    (DATA/'STOP_GENERATION').unlink(missing_ok=True)
    log=(OUT/'generated-overnight.log').open('w',encoding='utf-8')
    command=[sys.executable,'-u','generate_dataset.py','--size','384','--batch','4','--steps','2','--custom-target','5000','--real-target','100']
    process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    (DATA/'overnight-process.json').write_text(json.dumps({'pid':process.pid,'command':command,'targets':{'custom_per_class':5000,'real_per_class':100}}),encoding='utf-8',newline='\n')
    print('OVERNIGHT GENERATION RESUMED',process.pid,flush=True)
    run('report_generated_round.py',logname='generated-report.log')
