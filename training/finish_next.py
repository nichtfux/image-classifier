"""Background completion of the Python-only round. Never installs to a place."""
import ctypes,json,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent;OUT=HERE/'out'


def run(script,*args):
    print('RUN',script,*args,flush=True)
    subprocess.run([sys.executable,'-u',script,*args],cwd=HERE,check=True)


def main():
    record=json.loads((OUT/'gacl299_40_wide-process.json').read_text());pid=record['pid']
    while True:
        h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
        if not h:break
        code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h)
        if code.value!=259:break
        time.sleep(3)
    log=(OUT/'gacl299_40_wide-gacl.log').read_text()
    if 'DONE' not in log:raise RuntimeError('Training did not finish cleanly; inspect its log.')
    run('select_next.py')
    path=(OUT/'gacl299_40_wide-selected.txt').read_text().strip()
    run('export_candidate.py','--checkpoint',path)
    run('offline_native.py','--candidate',str(HERE.parent/'luau/models/gacl299_40_wide'),'--tag','gacl299_40_wide')
    run('evaluate_next.py')
    print('ROUND EVALUATION DONE',flush=True)


if __name__=='__main__':main()
