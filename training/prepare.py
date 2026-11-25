"""Render the downloaded strokes to 28x28 images with raster.py.

    python prepare.py   # -> ../data/rendered.npz  (+ ../data/test_strokes.json)

Train set: every drawing once clean and once augmented. Test set: the last
N_TEST recognised drawings of each category, clean.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from categories import REAL_CATEGORIES as CATEGORIES
import raster

DATA = os.path.join(HERE, "..", "data")
N_TRAIN = 20000
N_TEST = 2000


def load(name):
    out = []
    with open(os.path.join(DATA, "simplified", name + ".ndjson"), "rb") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("recognized") and d.get("drawing"):
                out.append(d["drawing"])
    return out


def job(args):
    ci, name = args
    rng = np.random.default_rng(1000 + ci)
    drawings = load(name)
    test = drawings[-N_TEST:]
    train = drawings[: min(N_TRAIN, len(drawings) - N_TEST)]
    xtr = np.empty((2 * len(train), 28, 28), np.uint8)
    for i, d in enumerate(train):
        xtr[2 * i] = np.round(raster.render(d) * 255)
        xtr[2 * i + 1] = np.round(raster.render(raster.augment(d, rng)) * 255)
    xte = np.stack([np.round(raster.render(d) * 255) for d in test]).astype(np.uint8)
    return ci, name, xtr, xte, test[:3], len(drawings)


def main():
    xs_tr, ys_tr, xs_te, ys_te, samples = [], [], [], [], []
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for ci, name, xtr, xte, few, n in pool.imap_unordered(job, list(enumerate(CATEGORIES))):
            print(f"{name:14s} recognised={n:6d} train={len(xtr)//2} test={len(xte)}", flush=True)
            xs_tr.append(xtr); ys_tr.append(np.full(len(xtr), ci, np.int64))
            xs_te.append(xte); ys_te.append(np.full(len(xte), ci, np.int64))
            for d in few:
                samples.append({"label": ci, "drawing": d})
    np.savez_compressed(os.path.join(DATA, "rendered.npz"),
                        xtr=np.concatenate(xs_tr), ytr=np.concatenate(ys_tr),
                        xte=np.concatenate(xs_te), yte=np.concatenate(ys_te))
    samples.sort(key=lambda s: s["label"])
    with open(os.path.join(DATA, "test_strokes.json"), "w") as f:
        json.dump(samples, f)


if __name__ == "__main__":
    main()
