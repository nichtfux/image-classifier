"""WMI-launched unattended handoff from the completed 40px run to 64px.

The running 40px supervisor and generator are observed only. This process owns
the later calibration, CPU indexing and 72-hour 64px training process tree.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
QUEUE=HERE/'out/round64'
OLD=HERE/'out/long299_40_v4'
RUN=HERE/'out/long299_64_v4'
STOP=QUEUE/'STOP'
STATUS=ROOT/'notes/long-run64-status.md'


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')
    for attempt in range(30):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            time.sleep(.1)
    raise RuntimeError('Cannot replace '+str(path))


def status(phase,worker=None):
    state={'phase':phase,'watcher_pid':os.getpid(),'worker_pid':worker,
           'updated_utc':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
           'hours':72,'stop_file':str(STOP),'auto_install':False}
    write_json(QUEUE/'state.json',state)
    if phase=='training':return
    details=["# Queued 64px training",'',f"Updated UTC {state['updated_utc']}",
             f"Phase {phase}",f"Watcher PID {os.getpid()}",
             'Sequence: finish 40px, validation-only calibration, CPU stroke index/cache, 72-hour 64px training.',
             'Architecture 48/96/192 convolution channels, dense 768, four 64px planes, 299 classes.',
             'Training retains 40% work duty and a bounded GPU allocator alongside generation.',
             f'Stop by creating `{STOP}`. Restart with `training/restart-round64.ps1`.',
             'Checkpoints include optimizer and RNG state. New generated v4 training rows reload every 15 minutes.',
             'Validation covers black, colour, random colour, filled and brush real/custom cohorts and false alarms.',
             'Exports and parity checks stay on disk for a later install decision.']
    if phase=='waiting for 40px completion':
        progress=OLD/'progress.json'
        try:
            old=json.loads(progress.read_text(encoding='utf-8'))
            remaining=max(0.,old['hours']*3600-old['elapsed'])
            eta=time.time()+remaining
            details += [f"40px elapsed {old['elapsed']/3600:.2f} hours, step {old['step']:,}.",
                        'Estimated handoff UTC '+time.strftime('%Y-%m-%d %H:%M',time.gmtime(eta))+'.',
                        'Calibration and CPU preparation precede GPU training.']
        except (OSError,ValueError,KeyError):
            details += ['Waiting for the 40px completion marker.']
    temporary=STATUS.with_suffix('.part.md')
    temporary.write_text('\n'.join(details)+'\n',encoding='utf-8',newline='\n')
    for attempt in range(30):
        try:
            os.replace(temporary,STATUS)
            break
        except PermissionError:
            time.sleep(.1)


def run_stage(phase,command):
    while not STOP.exists():
        with (QUEUE/'queue.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(command,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            status(phase,process.pid)
            while process.poll() is None:
                if STOP.exists() and phase=='training':
                    RUN.mkdir(parents=True,exist_ok=True)
                    (RUN/'STOP_LONG_RUN').touch()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            code=process.returncode
        if STOP.exists():
            status('stopped after the current checkpoint or preparation stage')
            return None
        if code in [0,75]:return code
        status(phase+' failed; retrying in 60 seconds')
        with (QUEUE/'queue.log').open('a',encoding='utf-8') as log:
            log.write(f'\nProcess exit {code}; retrying {phase}\n')
        for _ in range(60):
            if STOP.exists():return None
            time.sleep(1)
    return None


def trainer_active():
    path=RUN/'worker.lock'
    if not path.exists():return False
    import msvcrt
    with path.open('a+b') as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            return True
        handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
    return False


def main():
    QUEUE.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(QUEUE/'watcher.lock').open('a+b')
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    # Thread-local idle-sleep request, automatically released when this watcher exits.
    set_awake=ctypes.windll.kernel32.SetThreadExecutionState
    set_awake.argtypes=[ctypes.c_uint32]
    set_awake.restype=ctypes.c_uint32
    previous=set_awake(0x80000001)
    if previous==0:
        raise RuntimeError('Windows rejected the 64px keep-awake request')
    write_json(QUEUE/'awake.json',{'pid':os.getpid(),'system_required':True,
               'continuous':True,'previous_flags':int(previous),
               'covers':'waiting, calibration, CPU preparation and the entire 72-hour training process tree'})
    try:
        while not ((OLD/'DONE.json').exists() or (OLD/'HANDOFF_TO64.json').exists()):
            if STOP.exists():status('stopped while queued');return
            status('waiting for 40px completion')
            for _ in range(60):
                if STOP.exists():status('stopped while queued');return
                time.sleep(1)
        # DONE is written at the end of the trainer. Give its process a chance
        # to release CUDA allocations before the calibration process starts.
        time.sleep(5)
        # A replacement watcher can adopt a still-running child after a watcher
        # crash. It must not prepare or rewrite metadata under that trainer.
        while trainer_active():
            if STOP.exists():
                (RUN/'STOP_LONG_RUN').touch()
            status('training')
            time.sleep(10)
        if (RUN/'DONE.json').exists():status('64px run complete');return
        if run_stage('post-run 40px calibration',[sys.executable,'-u',str(HERE/'post40_calibration.py')]) is None:return
        if run_stage('CPU 64px data preparation',[sys.executable,'-u',str(HERE/'long_run_data64.py')]) is None:return
        while not (RUN/'DONE.json').exists() and not STOP.exists() and not (RUN/'STOP_LONG_RUN').exists():
            command=[sys.executable,'-u',str(HERE/'long_run64.py'),'--hours','72']
            if not (RUN/'first-leg-complete.json').exists():command.append('--first-leg')
            if run_stage('training',command) is None:return
        status('64px run complete' if (RUN/'DONE.json').exists() else '64px run stopped')
    finally:
        set_awake(0x80000000)


if __name__=='__main__':
    try:main()
    except Exception:
        QUEUE.mkdir(parents=True,exist_ok=True)
        (QUEUE/'watcher-error.txt').write_text(traceback.format_exc(),encoding='utf-8',newline='\n')
        raise
