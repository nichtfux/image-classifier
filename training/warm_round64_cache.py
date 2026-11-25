"""CPU cache preparation alongside the single 64px trainer's initial loading."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from colour import planes
from colour_raster import render
from long_run_data import rows
from long_run_data64 import FOLDER, GENERATED


def prepare(row):
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    cache=FOLDER/'generated'
    target=cache/(row['id']+'.npz')
    if target.exists():return False
    original=GENERATED/row['sample']
    assert hashlib.sha256(original.read_bytes()).hexdigest()==row['sample_sha256']
    geometry=json.loads((GENERATED/'samples'/(row['id']+'.json')).read_text(encoding='utf-8'))
    ink,owner,_=render(geometry['strokes'],64)
    x=np.rint(planes(ink.astype(np.float32)/255,owner,[c-1 for c in geometry['colours']])*255).astype(np.uint8)
    temporary=target.with_suffix(f'.warm-{os.getpid()}.npz')
    with np.load(original) as old:
        np.savez(temporary,x=x,y=old['y'],teacher=old['teacher_logits'])
    # Concurrent writers produce the same deterministic sample. Separate
    # temporary names prevent interference with the trainer's cache writer.
    if target.exists():
        temporary.unlink()
    else:
        try:temporary.replace(target)
        except PermissionError:
            if not target.exists():raise
            temporary.unlink()
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-index',type=int,default=18000)
    parser.add_argument('--end-index',type=int)
    args=parser.parse_args()
    selected=[]
    seen=set()
    for row in rows():
        if row['split']=='train' and row['input_sha256'] not in seen:
            seen.add(row['input_sha256']);selected.append(row)
    # The trainer is traversing from the beginning. Prepare the later portion.
    selected=selected[args.start_index:args.end_index]
    done=0
    with ProcessPoolExecutor(max_workers=3) as pool:
        for changed in pool.map(prepare,selected,chunksize=16):
            done+=1
            if done%1000==0:print('CPU_CACHE',done,len(selected),flush=True)
    print('CPU_CACHE_COMPLETE',done,flush=True)


if __name__=='__main__':main()
