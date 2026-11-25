"""Finish the warm checkpoint, then resume with the final augmentation revision."""
import ctypes,hashlib,json,os,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'out';checkpoint=OUT/'gacl299_40_wide-warm.pt'
pid=json.loads((OUT/'gacl299_40_wide-process.json').read_text())['pid']
started=time.time()
while not checkpoint.exists():
    if time.time()-started>3600:raise RuntimeError('warm checkpoint did not finish within one hour')
    h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not h:raise RuntimeError('training process exited before warm checkpoint')
    code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h)
    if code.value!=259:raise RuntimeError(f'training exited: {code.value}')
    time.sleep(1)
print('Warm checkpoint completed. Restarting own training process.',flush=True)
h=ctypes.windll.kernel32.OpenProcess(0x0001,False,pid)
if h:ctypes.windll.kernel32.TerminateProcess(h,0);ctypes.windll.kernel32.CloseHandle(h)
time.sleep(3)
progress=OUT/'gacl299_40_wide-progress.json'
if progress.exists():(OUT/'gacl299_40_wide-warm-progress.json').write_bytes(progress.read_bytes())
f=(OUT/'gacl299_40_wide-gacl.log').open('w',encoding='utf-8')
command=[sys.executable,'-u','train_next.py','--resume',str(checkpoint),'--hours','2.3','--epochs','3']
proc=subprocess.Popen(command,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
record={'pid':proc.pid,'command':command,'warm_checkpoint':str(checkpoint),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['train_next.py','gacl.py','style_augment.py','synth_diverse.py']}}
(OUT/'gacl299_40_wide-process.json').write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(record),flush=True)
