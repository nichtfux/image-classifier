"""Check corpus separation, original test preservation, and artifact label shapes."""
import json
from pathlib import Path
import numpy as np
from categories import CATEGORIES

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = DATA / "expanded"


def main():
    splits = json.loads((OUT / "splits.json").read_text())
    counts = json.loads((OUT / "counts.json").read_text())
    assert splits["categories"] == CATEGORIES
    for name in CATEGORIES:
        test, val = set(splits["test"][name]), set(splits["validation"][name])
        assert len(test) == 2000 and len(val) == 1000 and not (test & val)
    for split in ("train", "validation", "test"):
        x = np.load(OUT / (split + "-x.npy"), mmap_mode="r")
        y = np.load(OUT / (split + "-y.npy"), mmap_mode="r")
        assert x.shape == (len(y), 28, 28) and x.dtype == np.uint8
        assert np.array_equal(np.bincount(y, minlength=64), [counts[name][split] for name in CATEGORIES])
    # Compare every pixel, rather than assuming persisted IDs reproduced the split.
    with np.load(DATA / "rendered.npz") as old:
        ox, oy = old["xte"], old["yte"]
    nx = np.load(OUT / "test-x.npy", mmap_mode="r")
    for ci in range(32):
        assert np.array_equal(ox[oy == ci], nx[ci*2000:(ci+1)*2000]), CATEGORIES[ci]
    result = {"categories": len(CATEGORIES), "original_test_images_preserved": 64000,
              "test_validation_ids_disjoint": True,
              "train_drawings": sum(v["train"] for v in counts.values()),
              "validation_drawings": 64000, "test_drawings": 128000}
    (OUT / "audit.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
