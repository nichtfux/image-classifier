"""Background coordinator; preserves the original control run and live place."""
import argparse,ctypes,json,subprocess,sys,time,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'out';ROOT=HERE.parent/'data/generated'


def running(pid):
    h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not h:return False
    code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h)
    return code.value==259


def run(script,*args,logname=None):
    command=[sys.executable,'-u',script,*args]
    print('RUN',script,*args,flush=True)
    log=(OUT/(logname or script.replace('.py','.log'))).open('w',encoding='utf-8')
    process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    (ROOT/'round-current-process.json').write_text(json.dumps({'script':script,'pid':process.pid,'command':command}),encoding='utf-8',newline='\n')
    code=process.wait()
    if code:raise RuntimeError(f'{script} exited {code}; inspect {log.name}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--control-pid',type=int,default=0,help='Optional explicitly known unfinished control coordinator')
    ap.add_argument('--pilot-attempts',type=int,default=3000)
    ap.add_argument('--reuse-benchmark',action='store_true')
    ap.add_argument('--reuse-pilot',action='store_true')
    ap.add_argument('--reuse-trained',action='store_true')
    a=ap.parse_args()
    while not all((ROOT/(n+'-download.json')).exists() for n in ['flux1-schnell','siglip-so400m']):time.sleep(3)
    # Generic group-offload hooks are unsafe with this bitsandbytes/driver stack.
    # Preserve the running control and benchmark resident NF4 after it frees GPU.
    while a.control_pid and running(a.control_pid):time.sleep(3)
    if not (HERE.parent/'notes/training-round-next.json').exists():raise RuntimeError('Control evaluation missing')
    run('export_candidate.py','--checkpoint',str(OUT/'gacl299_40_wide.pt'),logname='control-final-export.log')
    run('report_next.py',logname='control-report.log')
    if not a.reuse_benchmark:run('benchmark_generator.py',logname='flux-resident-benchmark.log')
    results=json.loads((ROOT/'benchmark-resident.json').read_text())
    # Choose accepted samples/s, not raw images/s or resolution alone. A fast
    # setting whose drawings fail conversion/filtering is not useful training.
    choices=[r for r in results if r['kept']>0] or results
    best=max(choices,key=lambda r:(r['kept_per_second'],r['pipeline_images_per_second']))
    settings=['--size',str(best['size']),'--batch',str(best['batch']),'--steps',str(best['steps'])]
    pilot_attempts=max(a.pilot_attempts,(5*29+64)*best['batch'])
    if not a.reuse_pilot:run('generate_dataset.py',*settings,'--stop-after-attempts',str(pilot_attempts),logname='generated-pilot.log')
    if not a.reuse_trained:run('train_generated_ablation.py',logname='generated-ablation-train.log')
    for name in ['gacl299_40_continued','gacl299_40_genhard','gacl299_40_genkd']:
        run('export_candidate.py','--checkpoint',str(OUT/(name+'.pt')),'--name',name,logname=name+'-export.log')
        run('evaluate_next.py','--name',name,logname=name+'-evaluation.log')
        run('export_candidate.py','--checkpoint',str(OUT/(name+'.pt')),'--name',name,logname=name+'-final-export.log')
    run('evaluate_generated_holdout.py',logname='generated-holdout.log')
    run('offline_native.py','--candidate',str(HERE.parent/'luau/models/gacl299_40_genkd'),'--tag','gacl299_40_genkd',logname='generated-native.log')
    # Resume the large corpus only after the matched pilot experiments finish.
    log=(OUT/'generated-overnight.log').open('w',encoding='utf-8')
    command=[sys.executable,'-u','generate_dataset.py',*settings,'--custom-target','5000','--real-target','100']
    process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    (ROOT/'overnight-process.json').write_text(json.dumps({'pid':process.pid,'command':command,'targets':{'custom_per_class':5000,'real_per_class':100}}),encoding='utf-8',newline='\n')
    print('OVERNIGHT GENERATION STARTED',process.pid,flush=True)
    run('report_generated_round.py',logname='generated-report.log')


if __name__=='__main__':main()
