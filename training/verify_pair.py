"""Audit paired corpus IDs/order and every original-100 raster against its source."""
import hashlib
import json
from pathlib import Path
import numpy as np
from categories import REAL_CATEGORIES as CATEGORIES

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    dirs = [DATA / "expanded270-28", DATA / "expanded270-40"]
    manifests = [json.loads((p / "splits.json").read_text()) for p in dirs]
    assert manifests[0] == manifests[1], "paired split manifests differ"
    split = manifests[0]
    seed = json.loads((DATA / "expanded100/splits.json").read_text())
    assert split["categories"] == CATEGORIES and CATEGORIES[:100] == seed["categories"]
    counts = json.loads((dirs[0] / "counts.json").read_text())
    assert counts == json.loads((dirs[1] / "counts.json").read_text())
    shared = DATA / "expanded270-selections"
    preserved = {s: 0 for s in ("train", "validation", "test")}
    for ci, name in enumerate(CATEGORIES):
        test, val = set(split["test"][name]), set(split["validation"][name])
        assert len(test)==2000 and len(val)==1000 and not test & val
        ids = np.load(shared / f"{ci:03d}-train-ids.npy")
        assert len(ids) == counts[name]["train"]
        assert hashlib.sha256(ids.tobytes()).hexdigest() == counts[name]["train_ids_sha256"]
        assert not np.isin(ids, np.array([int(k) for k in test | val], dtype=np.uint64)).any()
        if ci < 100:
            for s in ("test", "validation"):
                assert set(split[s][name]) == set(seed[s][name]), (name, s)
            for s in ("train", "validation", "test"):
                old = np.load(DATA / "expanded100" / f"{ci:03d}-{s}.npy", mmap_mode="r")
                new = np.load(dirs[0] / f"{ci:03d}-{s}.npy", mmap_mode="r")
                assert np.array_equal(old, new), ("original raster changed", name, s)
                preserved[s] += len(old)
        if (ci+1) % 25 == 0:
            print(f"audited {ci+1}/{len(CATEGORIES)} categories", flush=True)
    for s in ("train", "validation", "test"):
        yy = [np.load(p / (s+"-y.npy"), mmap_mode="r") for p in dirs]
        assert yy[0].shape == yy[1].shape
        for start in range(0,len(yy[0]),1_000_000):
            assert np.array_equal(yy[0][start:start+1_000_000], yy[1][start:start+1_000_000]), (s,start)
        for p, size in zip(dirs,(28,40)):
            x = np.load(p/(s+"-x.npy"),mmap_mode="r")
            assert x.shape == (len(yy[0]),size,size) and x.dtype==np.uint8
        assert np.array_equal(np.bincount(yy[0],minlength=270),[counts[n][s] for n in CATEGORIES])
    report={"categories":270,"same_train_validation_test_ids":True,"same_combined_row_order":True,
            "original100_images_preserved_pixel_for_pixel":preserved,
            "train_drawings":sum(v["train"] for v in counts.values()),"validation_drawings":270000,"test_drawings":540000}
    (shared/"audit.json").write_text(json.dumps(report,indent=1),encoding="utf-8",newline="\n")
    print(json.dumps(report,indent=1))


if __name__ == "__main__":
    main()
