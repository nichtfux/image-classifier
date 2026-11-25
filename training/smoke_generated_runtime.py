"""Exercise complex converted training gestures in dev-only Luau CLI runtimes."""
import json,subprocess,sys
from pathlib import Path
from generated_splits import assign
from candidate_runtime import build
from offline_native import run

ROOT=Path(__file__).resolve().parent.parent

if __name__=='__main__':
    data=ROOT/'data/generated';content=(data/'manifest.jsonl').read_text(encoding='utf-8')
    lines=content.splitlines()
    if not content.endswith('\n'):lines=lines[:-1]
    rows=[json.loads(line) for line in lines if line.strip()]
    rows=[r for r in rows if r['keep'] and assign(r)=='train']
    selected=sorted(rows,key=lambda r:r['point_count'],reverse=True)[:3]
    for style in ['marker','outline','flat','painted','kids']:
        row=next((r for r in rows if r['style']==style and r not in selected),None)
        if row:selected.append(row)
    vectors=[]
    for r in selected:
        g=json.loads((data/'samples'/(r['id']+'.json')).read_text())
        vectors.append({'label':r['label'],'drawing':g['strokes'],'colours':[v-1 for v in g['colours']]})
    path=data/'runtime-smoke-inputs.json';path.write_text(json.dumps(vectors),encoding='utf-8',newline='\n')
    folder=ROOT/'training/out/complex-runtime-smoke'
    subprocess.run([sys.executable,'export.py','--name','gacl299_40_wide','--out-dir',str(folder),'--samples','1','--probes','--extra-vectors',str(path)],cwd=ROOT/'training',check=True)
    build(folder)
    result=run(folder,'generated-complex-smoke')
    (data/'generated-runtime-smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print('COMPLEX GENERATED RUNTIME PASS',flush=True)
