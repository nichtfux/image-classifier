"""Final native/interpreted parity for the two unchanged matched branches."""
import json
from pathlib import Path
from offline_native import run
ROOT=Path(__file__).resolve().parent.parent
if __name__=='__main__':
    for name in ['gacl299_40_continued','gacl299_40_genhard']:
        result=run(ROOT/'luau/models'/name,name+'-final')
        (ROOT/'notes'/(name+'-runtime-parity.json')).write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print('ALL MATCHED CANDIDATE PARITY PASSED',flush=True)
