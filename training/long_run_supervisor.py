"""Detached crash/restart supervisor. Does not start/stop the generation worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
RUN=HERE/'out/long299_40_v4'


def supervise():
    RUN.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(RUN/'supervisor.lock').open('a+b');lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    failures=0
    while not (RUN/'STOP_LONG_RUN').exists() and not (RUN/'DONE.json').exists():
        first=not (RUN/'first-leg-complete.json').exists()
        command=[sys.executable,'-u',str(HERE/'long_run.py'),'--hours','60']
        if first:command.append('--first-leg')
        with (RUN/'train.log').open('ab',buffering=0) as log:
            worker=subprocess.Popen(command,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            (RUN/'processes.json').write_text(json.dumps({'supervisor_pid':os.getpid(),'worker_launcher_pid':worker.pid,
                                  'command':command,'started_utc':time.time()},indent=2),encoding='utf-8',newline='\n')
            code=worker.wait()
        if code==0:return
        if code==75:
            print('First checkpoint complete. Restarting from optimizer/RNG checkpoint.',flush=True)
            continue
        failures+=1
        print('Training exited',code,'retry',failures,'in 60 seconds; inspect last-error.txt',flush=True)
        for _ in range(60):
            if (RUN/'STOP_LONG_RUN').exists():return
            time.sleep(1)


def launch():
    RUN.mkdir(parents=True,exist_ok=True)
    # Break away from the invoking terminal/job object. This process tree has no
    # stdin, inherited console, or pipe connected to a Codex/Claude tool lifetime.
    with (RUN/'supervisor.log').open('ab',buffering=0) as log:
        flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
        worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--supervise'],cwd=HERE,
                  stdin=subprocess.DEVNULL,stdout=log,stderr=log,close_fds=True,creationflags=flags)
    print('Detached supervisor launcher PID',worker.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--supervise',action='store_true')
    if parser.parse_args().supervise:supervise()
    else:launch()
