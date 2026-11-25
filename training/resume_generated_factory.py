"""Cleanly reload the long generation worker after a pipeline update."""
import json,subprocess,sys
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'data/generated'
if __name__=='__main__':
    old=json.loads((ROOT/'overnight-process.json').read_text())
    try:psutil.Process(old['pid']).wait()
    except psutil.NoSuchProcess:pass
    (ROOT/'STOP_GENERATION').unlink(missing_ok=True)
    command=[sys.executable,'-u','generate_dataset.py','--size','384','--batch','4','--steps','2','--custom-target','5000','--real-target','100']
    log=(HERE/'out/generated-overnight.log').open('w',encoding='utf-8')
    p=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    (ROOT/'overnight-process.json').write_text(json.dumps({'pid':p.pid,'command':command,'targets':{'custom_per_class':5000,'real_per_class':100}}),encoding='utf-8',newline='\n')
    print('FACTORY RESUMED',p.pid,flush=True)
