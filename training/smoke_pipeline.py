"""Exercise disk chunks, both input shapes, quantization and export on tiny data."""
import contextlib
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import train
import export
from categories import CATEGORIES

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "data"
    with tempfile.TemporaryDirectory(prefix="pipeline-smoke-", dir=base) as temp:
        root = Path(temp)
        train.DATA, train.OUT = str(root), str(root / "out")
        export.HERE, export.DATA = str(root), str(root)
        ncls = len(CATEGORIES)
        for size in (28, 40):
            corpus = root / f"smoke{size}"
            corpus.mkdir()
            rng = np.random.default_rng(7)
            for split, n in (("train", 2*ncls), ("validation", ncls), ("test", 2*ncls)):
                np.save(corpus / (split+"-x.npy"), rng.integers(0,256,(n,size,size),dtype=np.uint8))
                np.save(corpus / (split+"-y.npy"), np.arange(n,dtype=np.int64)%ncls)
            (corpus/"splits.json").write_text(json.dumps({"categories":CATEGORIES}))
            (corpus/"corpus.json").write_text(json.dumps({"size":size,"training_order":"interleaved256","train_ids_sha256":"smoke"}))
            (corpus/"test_strokes.json").write_text(json.dumps([{"label":0,"drawing":[[[0,100,50],[0,0,100]]]}]))
            name=f"smoke{size}"
            sys.argv=["train.py","--corpus",corpus.name,"--name",name,"--widths","3,6,9","--fc","12", "--size",str(size),
                      "--epochs","1","--chunk","128","--bs","64","--eval-bs","128","--clean"]
            with contextlib.redirect_stdout(io.StringIO()):
                train.main()
            with np.load(root/"out"/(name+".npz")) as weights:
                assert weights["w3"].shape==(12,9*(size//8)**2)
                assert weights["w4"].shape==(ncls,12)
            sys.argv=["export.py","--name",name,"--samples","1","--out-dir",str(root/f"export{size}")]
            with contextlib.redirect_stdout(io.StringIO()):
                export.main()
            assert f"size = {size}," in (root/f"export{size}/Weights.luau").read_text()
            assert f"size = {size}," in (root/f"export{size}/TestVectors.luau").read_text()
            gc.collect()
            print(f"{size}px: memmap/chunks, dynamic dense shape, quantization and export passed", flush=True)
    print("Temporary smoke data removed.")
