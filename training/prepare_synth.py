"""Append hand-reviewed synthetic strokes to the unchanged 270-class 28px corpus.

The real per-class arrays are reused read-only; their split IDs and pixels remain
identical. Synthetic IDs are strings, so do not coerce them to Quick, Draw! uint64.
Nothing here displays drawings or sends strokes to Studio or a server.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from categories import CATEGORIES
from fast_raster import flatten, render_flat
from prepare_pair import write_json
from synth_doodles import THINGS

DATA = Path(__file__).resolve().parent.parent / "data"


def stroke_hash(row):
    return hashlib.sha256(json.dumps(row["drawing"], separators=(",", ":")).encode()).hexdigest()


def build_synth(ci, name, source, out):
    raw = source / (name + ".ndjson")
    identity = {"source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "name": name, "size": 28, "validation": 1000, "test": 2000}
    marker = out / f"{ci:03d}-done.json"
    files = [out / f"{ci:03d}-{split}.npy" for split in ("train", "validation", "test")]
    if marker.exists():
        cached = json.loads(marker.read_text())
        if cached["identity"] == identity and all(p.exists() for p in files):
            return cached
    rows = [json.loads(line) for line in raw.open(encoding="utf-8")]
    assert len(rows) == 30000, (name, len(rows))
    assert all(r["word"] == name and r.get("recognized") and r["drawing"] for r in rows)
    ids = [str(r["key_id"]) for r in rows]
    assert len(ids) == len(set(ids)), name
    train, validation, test = rows[:-3000], rows[-3000:-2000], rows[-2000:]
    held_hashes = {stroke_hash(r) for r in validation + test}
    assert len(held_hashes) == 3000, f"duplicate held-out strokes in {name}"
    seen, unique = set(), []
    for r in train:
        h = stroke_hash(r)
        if h not in held_hashes and h not in seen:
            unique.append(r); seen.add(h)
    train = unique
    for split, selected in (("train", train), ("validation", validation), ("test", test)):
        arr = np.lib.format.open_memmap(out / f"{ci:03d}-{split}.npy", mode="w+",
                                       dtype=np.uint8, shape=(len(selected), 28, 28))
        for i, r in enumerate(selected):
            arr[i] = render_flat(*flatten(r["drawing"]), 28, 22.0, 1.0)
        arr.flush(); del arr
    result = {"identity": identity, "name": name, "train": len(train),
              "validation": 1000, "test": 2000,
              "train_ids_sha256": hashlib.sha256("\n".join(str(r["key_id"]) for r in train).encode()).hexdigest(),
              "validation_ids": [str(r["key_id"]) for r in validation],
              "test_ids": [str(r["key_id"]) for r in test],
              "samples": [{"label": ci, "drawing": r["drawing"]} for r in test[:3]]}
    write_json(marker, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="expanded270-28")
    ap.add_argument("--synth", default="synth")
    ap.add_argument("--out", default="expanded299-28")
    args = ap.parse_args()
    base, source, out = DATA / args.real, DATA / args.synth, DATA / args.out
    out.mkdir(exist_ok=True)
    seed = json.loads((base / "splits.json").read_text())
    assert seed["categories"] == CATEGORIES[:270] and CATEGORIES[270:] == list(THINGS)
    counts = json.loads((base / "counts.json").read_text())
    samples = json.loads((base / "test_strokes.json").read_text())
    splits = {**seed, "categories": CATEGORIES, "test": dict(seed["test"]),
              "validation": dict(seed["validation"])}
    t0 = time.time()
    sources, source_hashes = [], {}
    for ci, name in enumerate(CATEGORIES):
        if ci < 270:
            sources.append(base)
        else:
            result = build_synth(ci, name, source, out)
            source_hashes[name] = result["identity"]["source_sha256"]
            counts[name] = {k: result[k] for k in ("name", "train", "validation", "test", "train_ids_sha256")}
            splits["test"][name], splits["validation"][name] = result["test_ids"], result["validation_ids"]
            samples.extend(result["samples"])
            sources.append(out)
            print(name, counts[name]["train"], "train", flush=True)
    # Source arrays avoid re-rendering or re-selecting any real drawing.
    audit = {"real_categories": 270, "synthetic_categories": 29,
             "real_splits_identical": all(splits[s][n] == seed[s][n]
                                          for s in ("test", "validation") for n in CATEGORIES[:270]),
             "real_per_class_arrays_reused": True, "real_heldout_pixels_identical": {}}
    for split in ("train", "validation", "test"):
        total = sum(counts[n][split] for n in CATEGORIES)
        arrays = [np.load(root / f"{ci:03d}-{split}.npy", mmap_mode="r") for ci, root in enumerate(sources)]
        x = np.lib.format.open_memmap(out / f"{split}-x.npy", mode="w+", dtype=np.uint8, shape=(total, 28, 28))
        y = np.lib.format.open_memmap(out / f"{split}-y.npy", mode="w+", dtype=np.int64, shape=(total,))
        off = 0
        if split == "train":
            rounds = (max(map(len, arrays)) + 255) // 256
            for ri in range(rounds):
                for ci, a in enumerate(arrays):
                    block = a[len(a)*ri//rounds:len(a)*(ri+1)//rounds]
                    x[off:off+len(block)], y[off:off+len(block)] = block, ci
                    off += len(block)
        else:
            for ci, a in enumerate(arrays):
                x[off:off+len(a)], y[off:off+len(a)] = a, ci
                off += len(a)
            # Audit every old held-out pixel and label against the shipped corpus.
            oldx = np.load(base / f"{split}-x.npy", mmap_mode="r")
            oldy = np.load(base / f"{split}-y.npy", mmap_mode="r")
            equal = all(np.array_equal(x[i:min(i+4096,len(oldx))], oldx[i:min(i+4096,len(oldx))])
                        for i in range(0, len(oldx), 4096))
            assert equal and np.array_equal(y[:len(oldy)], oldy), split
            audit["real_heldout_pixels_identical"][split] = True
            del oldx, oldy
        assert off == total
        x.flush(); y.flush(); del x, y, arrays
        print(split, total, "combined", round(time.time()-t0), "seconds", flush=True)
    write_json(out / "splits.json", splits)
    write_json(out / "counts.json", counts)
    write_json(out / "test_strokes.json", samples)
    write_json(out / "audit.json", audit)
    membership = hashlib.sha256("".join(counts[n]["train_ids_sha256"] for n in CATEGORIES).encode()).hexdigest()
    write_json(out / "corpus.json", {"size": 28, "box": 22.0, "radius": 1.0,
               "training_order": "interleaved256", "train_ids_sha256": membership,
               "seed_splits": args.real, "real_classes": 270, "synthetic_classes": 29,
               "real_train_ids_sha256": json.loads((base / "corpus.json").read_text())["train_ids_sha256"],
               "synthetic_generator_sha256": hashlib.sha256(Path(__file__).with_name("synth_doodles.py").read_bytes()).hexdigest(),
               "synthetic_source_sha256": source_hashes,
               "review_provenance": "Initial 157 ratings (150 yes, 7 no) informed the generators. The author approved reworking bomb/black hole on 2026-11-09; Claude supplied both at seed 2026. A bomb pose coverage correction then added top-mounted caps alongside diagonal caps and protected cap/fuse strokes; no mouse examples were added to training.",
               "synthetic_test_caveat": "Same generator families as training: optimistic upper bound, not real-player accuracy."})
    print("299-class 28px corpus ready", flush=True)


if __name__ == "__main__":
    main()
