"""Prompt-disjoint generated splits, independent of teacher acceptance.

All random seeds for the same class/prompt stay together. The frozen pilot ID
snapshot, not any legacy seed-based split field, is authoritative for training.
"""
import hashlib,json,os
from pathlib import Path
import numpy as np

METHOD='class-prompt-sha256-v1'


def group_key(row):
    return hashlib.sha256(f"{row['label']}|{row['prompt']}|{METHOD}".encode()).hexdigest()


def assign(row):
    bucket=int(group_key(row)[:8],16)%10
    return 'test' if bucket==0 else 'validation' if bucket==1 else 'train'


def finalize(root):
    """Only call between generation and training; acquire the writer's lock."""
    lock=(root/'generation.lock').open('a+b');lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        from generated_convert import validate_geometry
        seen=set();duplicates=0;unsafe=0
        for row in rows:
            if row['keep']:
                geometry=json.loads((root/'samples'/(row['id']+'.json')).read_text())
                try:row['work_length_to_span']=validate_geometry(geometry['strokes'])
                except ValueError as error:
                    row['keep']=False;row['reason']='conversion: '+str(error);unsafe+=1;continue
                with np.load(root/row['sample']) as z:signature=hashlib.sha256(z['x'].tobytes()).hexdigest()
                row['input_sha256']=signature
                if signature in seen:
                    row['keep']=False;row['reason']='duplicate_converted_input';duplicates+=1;continue
                seen.add(signature)
                row['split']=assign(row);row['split_method']=METHOD
                row['split_group']=group_key(row)
        path=root/'manifest.jsonl';temp=path.with_suffix('.split-part')
        temp.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows),encoding='utf-8',newline='\n')
        os.replace(temp,path)
        status_path=root/'generation-status.json';status=json.loads(status_path.read_text())
        status['duplicates_removed_at_freeze']=duplicates;status['illegal_geometry_removed_at_freeze']=unsafe
        status['kept']=sum(r['keep'] for r in rows)
        session=rows[-status['session_attempted']:] if status['session_attempted'] else []
        status['session_kept']=sum(r['keep'] for r in session)
        status['kept_per_second']=status['session_kept']/max(status['session_seconds'],1e-9)
        for ci,entry in enumerate(status['classes']):
            cohort=[r for r in rows if r['label']==ci];entry['kept']=sum(r['keep'] for r in cohort)
            entry['keep_rate']=entry['kept']/entry['attempted'] if entry['attempted'] else None
            for style,counts in entry['styles'].items():counts['kept']=sum(r['keep'] for r in cohort if r['style']==style)
        status_path.write_text(json.dumps(status,indent=2),encoding='utf-8',newline='\n')
        return rows
    finally:lock.close()
