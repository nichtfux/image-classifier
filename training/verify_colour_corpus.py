"""Audit both full corpora's pinned row order and stroke metadata in bounded reads."""
import json
from pathlib import Path
import numpy as np
from colour_data import ColourData

if __name__=='__main__':
    result={}
    for split in ('train','validation','test'):
        a,b=ColourData(split),ColourData(split,'colour299-40')
        assert np.array_equal(a.y,b.y)
        for lo in range(0,len(a.offset),1000000):
            assert np.array_equal(a.offset[lo:lo+1000000],b.offset[lo:lo+1000000])
        for lo in range(0,len(a.length),1000000):
            assert np.array_equal(a.length[lo:lo+1000000],b.length[lo:lo+1000000])
        assert a.x.shape==(len(a.y),28,28) and b.x.shape==(len(b.y),40,40)
        result[split]={'drawings':len(a.y),'strokes':len(a.length),'row_labels_equal':True,'offsets_lengths_exact':True}
    p=Path(__file__).resolve().parent.parent/'notes/colour-corpus-audit.json'
    p.write_text(json.dumps(result,indent=1),encoding='utf-8',newline='\n');print(json.dumps(result))
