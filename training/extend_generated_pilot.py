"""Expand the first generated pilot before the waiting coordinator trains it.

Only the known research coordinator is paused; the generator continues normally.
This utility always resumes the coordinator, including after a failed expansion.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'data/generated'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--coordinator',type=int,required=True)
    ap.add_argument('--generator',type=int,required=True);ap.add_argument('--total',type=int,default=3000)
    a=ap.parse_args();owner=psutil.Process(a.coordinator)
    if not any(Path(v).name=='run_generated_round.py' for v in owner.cmdline()):raise RuntimeError('Wrong coordinator')
    owner.suspend()
    try:
        print('Paused coordinator until expanded pilot is ready',flush=True)
        try:psutil.Process(a.generator).wait()
        except psutil.NoSuchProcess:pass
        status=json.loads((ROOT/'generation-status.json').read_text())
        remaining=a.total-status['attempted']
        if remaining>0:
            command=[sys.executable,'-u','generate_dataset.py','--size','384','--batch','4','--steps','2','--stop-after-attempts',str(remaining)]
            log=(HERE/'out/generated-pilot-expansion.log').open('w',encoding='utf-8')
            p=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            (ROOT/'pilot-expansion-process.json').write_text(json.dumps({'pid':p.pid,'command':command,'target_total_attempts':a.total}),encoding='utf-8',newline='\n')
            if p.wait():raise RuntimeError('Pilot expansion failed; inspect log')
        print('Expanded pilot ready',flush=True)
    finally:
        owner.resume();print('Resumed research coordinator',flush=True)


if __name__=='__main__':main()
