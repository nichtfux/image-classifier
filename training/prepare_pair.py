"""Build paired 28/40 corpora from the same selected drawings, without RAM copies.

Per-category completion records allow restart. The training arrays interleave 256
rows from each category, so shuffled contiguous GPU chunks contain mixed labels
and can stream sequentially from disk instead of faulting random 4KB pages.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import time
import numpy as np
from categories import REAL_CATEGORIES as CATEGORIES
from expand_data import DATA, RAW, N_TEST, N_VAL, records
from fast_raster import flatten, render_flat


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=1), encoding="utf-8", newline="\n")
    temp.replace(path)


def valid_records(name):
    for row in records(name):
        if any(len(s[0]) for s in row["drawing"]):
            yield row


def category_job(args):
    ci, name, test_ids, val_ids, prefix, n_train, sizes = args
    shared = DATA / (prefix + "-selections")
    marker = shared / f"{ci:03d}-done.json"
    raw_stat = (RAW / (name + ".ndjson")).stat()
    identity = {"name": name, "n_train_limit": n_train, "sizes": sizes,
                "raw_bytes": raw_stat.st_size, "raw_mtime_ns": raw_stat.st_mtime_ns}
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["identity"] == identity and (not test_ids or set(prior["test_ids"]) == test_ids) \
                and (not val_ids or set(prior["validation_ids"]) == val_ids):
            files = [DATA / f"{prefix}-{size}" / f"{ci:03d}-{split}.npy" for size in sizes for split in ("train", "validation", "test")]
            if all(p.exists() for p in files):
                return prior
    if not test_ids or not val_ids:
        ids = [str(d["key_id"]) for d in valid_records(name)]
        if not test_ids:
            test_ids = set(ids[-N_TEST:])
        if not val_ids:
            val_ids = set([key for key in ids if key not in test_ids][-N_VAL:])
    assert len(test_ids) == N_TEST and len(val_ids) == N_VAL and not test_ids & val_ids, name
    train, val, test = [], [], []
    for d in valid_records(name):
        key = str(d["key_id"])
        if key in test_ids:
            test.append(d)
        elif key in val_ids:
            val.append(d)
        elif len(train) < n_train:
            train.append(d)
    assert len(test) == N_TEST and len(val) == N_VAL, name
    def digest(d):
        return hashlib.blake2s(json.dumps(d["drawing"], separators=(",", ":")).encode()).digest()
    held_hashes, seen = {digest(d) for d in val + test}, set()
    unique = []
    for d in train:
        h = digest(d)
        if h not in seen and h not in held_hashes:
            seen.add(h)
            unique.append(d)
    train = unique
    ids = np.array([int(d["key_id"]) for d in train], dtype="<u8")
    assert not set(map(str, ids.tolist())) & (test_ids | val_ids)
    np.save(shared / f"{ci:03d}-train-ids.npy", ids)
    for split, rows in (("train", train), ("validation", val), ("test", test)):
        outputs = [np.lib.format.open_memmap(DATA / f"{prefix}-{size}" / f"{ci:03d}-{split}.npy",
                   mode="w+", dtype=np.uint8, shape=(len(rows), size, size)) for size in sizes]
        for i, d in enumerate(rows):
            flat = flatten(d["drawing"])
            for size, arr in zip(sizes, outputs):
                arr[i] = render_flat(*flat, size, 22.0 * size / 28, size / 28)
        for arr in outputs:
            arr.flush()
        del arr, outputs
    result = {"identity": identity, "name": name, "train": len(train), "validation": len(val), "test": len(test),
              "train_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
              "test_ids": sorted(test_ids), "validation_ids": sorted(val_ids),
              "samples": [{"label": ci, "drawing": d["drawing"]} for d in test[:3]]}
    write_json(marker, result)
    return result


def combine(prefix, size, counts):
    out = DATA / f"{prefix}-{size}"
    for split in ("train", "validation", "test"):
        total = sum(counts[name][split] for name in CATEGORIES)
        x = np.lib.format.open_memmap(out / (split + "-x.npy"), mode="w+", dtype=np.uint8, shape=(total, size, size))
        y = np.lib.format.open_memmap(out / (split + "-y.npy"), mode="w+", dtype=np.int64, shape=(total,))
        arrays = [np.load(out / f"{ci:03d}-{split}.npy", mmap_mode="r") for ci in range(len(CATEGORIES))]
        off = 0
        if split == "train":
            rounds = (max(len(a) for a in arrays) + 255) // 256
            for ri in range(rounds):
                for ci, a in enumerate(arrays):
                    # Proportional slices keep smaller categories present in every
                    # round, instead of exhausting them before the end of the file.
                    block = a[len(a)*ri//rounds:len(a)*(ri+1)//rounds]
                    x[off:off+len(block)] = block
                    y[off:off+len(block)] = ci
                    off += len(block)
        else:
            for ci, a in enumerate(arrays):
                x[off:off+len(a)] = a
                y[off:off+len(a)] = ci
                off += len(a)
        assert off == total
        x.flush(); y.flush()
        del arrays, x, y
        print(f"combined {size}px {split}: {total:,} drawings", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="expanded270")
    ap.add_argument("--seed-splits", default="expanded100")
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sizes", default="28,40")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    assert sizes == [28, 40], "this paired comparison is specifically 28px and 40px"
    shared = DATA / (args.out + "-selections")
    shared.mkdir(exist_ok=True)
    for size in sizes:
        (DATA / f"{args.out}-{size}").mkdir(exist_ok=True)
    seed = json.loads((DATA / args.seed_splits / "splits.json").read_text())
    assert seed["categories"] == CATEGORIES[:len(seed["categories"])], "labels must stay append-only"
    manifest = shared / "splits.json"
    splits = json.loads(manifest.read_text()) if manifest.exists() else seed
    for name in seed["categories"]:
        for split in ("test", "validation"):
            assert set(splits[split][name]) == set(seed[split][name]), (split, name)
    splits["categories"] = CATEGORIES
    write_json(manifest, splits)
    order = sorted(range(len(CATEGORIES)), key=lambda ci: -(RAW / (CATEGORIES[ci] + ".ndjson")).stat().st_size)
    jobs = [(ci, CATEGORIES[ci], set(splits["test"].get(CATEGORIES[ci], [])),
             set(splits["validation"].get(CATEGORIES[ci], [])), args.out, args.n_train, sizes) for ci in order]
    counts, samples = {}, []
    t0 = time.time()
    # Retire workers periodically: parsing large nested JSON lists can leave
    # fragmented Python arenas resident even after a category's lists are freed.
    with Pool(args.workers, maxtasksperchild=3) as pool:
        for result in pool.imap_unordered(category_job, jobs):
            name = result["name"]
            counts[name] = {key: result[key] for key in ("name", "train", "validation", "test", "train_ids_sha256")}
            samples.extend(result["samples"])
            splits["test"][name], splits["validation"][name] = result["test_ids"], result["validation_ids"]
            if len(counts) % 16 == 0:
                write_json(manifest, splits)
            print(f"{len(counts):3d}/{len(CATEGORIES)} {name}: train {result['train']:,} ({time.time()-t0:.0f}s)", flush=True)
    write_json(manifest, splits)
    samples.sort(key=lambda s: s["label"])
    write_json(shared / "counts.json", counts)
    write_json(shared / "test_strokes.json", samples)
    membership_hash = hashlib.sha256("".join(counts[name]["train_ids_sha256"] for name in CATEGORIES).encode()).hexdigest()
    for size in sizes:
        out = DATA / f"{args.out}-{size}"
        for filename in ("splits.json", "counts.json", "test_strokes.json"):
            (out / filename).write_bytes((shared / filename).read_bytes())
        write_json(out / "corpus.json", {"size": size, "box": 22.0*size/28, "radius": size/28,
                   "selection_dir": shared.name, "training_order": "interleaved256",
                   "train_ids_sha256": membership_hash, "n_train_limit": args.n_train,
                   "seed_splits": args.seed_splits, "renderer": "fast_raster float64, no fast-math"})
        combine(args.out, size, counts)
    print("paired corpora complete", membership_hash, flush=True)


if __name__ == "__main__":
    main()
