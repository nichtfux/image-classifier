"""Build an expanded corpus, retaining the original model's exact test drawings.

Run before expanding the raw downloads: python expand_data.py --pin-original
Then python expand_data.py --download and python expand_data.py --prepare.
Only recognised, unique drawings are used; validation and test IDs are persisted.

Later rounds write a separate corpus folder, seeded from an earlier corpus's split
manifest so every earlier test/validation drawing stays held out, e.g. the
100-category round:
    python expand_data.py --download
    python expand_data.py --prepare --out expanded100 --seed-splits expanded --n-train 100000
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
from categories import REAL_CATEGORIES as CATEGORIES
import raster

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "simplified"
N_VAL, N_TEST = 1000, 2000
DOWNLOAD_BYTES = 64 * 1024 * 1024


def records(name):
    with (RAW / (name + ".ndjson")).open("rb") as f:
        for line in f:
            d = json.loads(line)
            if d.get("recognized") and d.get("drawing"):
                yield d


def pin_original(out):
    manifest = out / "splits.json"
    out.mkdir(exist_ok=True)
    if manifest.exists():
        print("split manifest already exists; preserving it", flush=True)
        return
    assert all((RAW / (name + ".ndjson")).stat().st_size <= 16 * 1024 * 1024 for name in CATEGORIES[:32]), \
        "original test IDs must be pinned from the original 16 MiB downloads; preserve the existing splits.json"
    splits = {"categories": CATEGORIES[:32], "seed": 20261109, "test": {}, "validation": {}}
    for name in CATEGORIES[:32]:
        ids = [str(d["key_id"]) for d in records(name)]
        splits["test"][name] = ids[-N_TEST:]
    manifest.write_text(json.dumps(splits, indent=1), encoding="utf-8", newline="\n")
    print("pinned original 64,000 test IDs", flush=True)


def download(name):
    # Files are downloaded once (the first DOWNLOAD_BYTES, or the whole file if smaller)
    # and never re-fetched, so earlier rounds' raw data stays exactly as it was.
    path = RAW / (name + ".ndjson")
    if path.exists():
        return name, "cached"
    url = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/" + urllib.parse.quote(name) + ".ndjson"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{DOWNLOAD_BYTES - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read(DOWNLOAD_BYTES)
    data = data[:data.rfind(b"\n") + 1]
    temp = path.with_suffix(".partial")
    temp.write_bytes(data)
    temp.replace(path)
    return name, f"{len(data) // 1024} KB"


def prepare_category(args):
    # Runs in a spawned worker (Windows): everything it needs comes in through args.
    ci, name, test_ids, val_ids, out, n_train = args
    out = Path(out)
    if not test_ids or not val_ids:
        ids = [str(d["key_id"]) for d in records(name)]
        if not test_ids:
            test_ids = set(ids[-N_TEST:])
        eligible = [key for key in ids if key not in test_ids]
        if not val_ids:
            val_ids = set(eligible[-N_VAL:])
    # Keep at most the requested number of training candidates in memory. Selection
    # follows file order, matching the original pipeline, with all held-outs excluded.
    held = set(test_ids) | set(val_ids)
    train, val, test = [], [], []
    seen = set()
    for d in records(name):
        key = str(d["key_id"])
        if key in test_ids:
            test.append(d)
        elif key in val_ids:
            val.append(d)
        elif key not in held and len(train) < n_train:
            train.append(d)
    assert len(test) == N_TEST and len(val) == N_VAL, (name, len(test), len(val))
    # Exclude exact duplicate strokes from training when they also occur held-out.
    def digest(d):
        return hashlib.blake2s(json.dumps(d["drawing"], separators=(",", ":")).encode()).digest()
    held_hashes = {digest(d) for d in val + test}
    unique = []
    for d in train:
        h = digest(d)
        if h not in seen and h not in held_hashes:
            unique.append(d)
            seen.add(h)
    train = unique
    for split, rows in (("train", train), ("validation", val), ("test", test)):
        arr = np.lib.format.open_memmap(out / f"{ci:03d}-{split}.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 28, 28))
        for i, d in enumerate(rows):
            arr[i] = np.round(raster.render(d["drawing"]) * 255)
        del arr
    samples = [{"label": ci, "drawing": d["drawing"]} for d in test[:3]]
    return {"name": name, "train": len(train), "validation": len(val), "test": len(test)}, samples, sorted(test_ids), sorted(val_ids)


def prepare(out, n_train, seed_splits):
    manifest = out / "splits.json"
    out.mkdir(exist_ok=True)
    if not manifest.exists():
        assert seed_splits, "no split manifest in " + str(out) + "; pass --seed-splits <earlier corpus>"
        manifest.write_text((DATA / seed_splits / "splits.json").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        print("seeded split manifest from", seed_splits, flush=True)
    splits = json.loads(manifest.read_text(encoding="utf-8"))
    old = splits["categories"]
    assert CATEGORIES[:len(old)] == old, "category order changed: categories are append-only"
    splits["categories"] = CATEGORIES
    # Discover each category's split inside its worker too. Scanning gigabytes
    # serially before starting the render pool would leave most CPU cores idle.
    # Biggest raw files first, so the slowest workers start early.
    order = sorted(range(len(CATEGORIES)), key=lambda ci: -(RAW / (CATEGORIES[ci] + ".ndjson")).stat().st_size)
    jobs = [(ci, CATEGORIES[ci], set(splits["test"].get(CATEGORIES[ci], [])), set(splits["validation"].get(CATEGORIES[ci], [])),
             str(out), n_train) for ci in order]
    counts, samples = {}, []
    with Pool(min(22, max(1, os.cpu_count() - 2))) as pool:
        for count, few, test_ids, val_ids in pool.imap_unordered(prepare_category, jobs):
            counts[count["name"]] = count
            samples.extend(few)
            splits["test"][count["name"]] = test_ids
            splits["validation"][count["name"]] = val_ids
            print(count, flush=True)
    manifest.write_text(json.dumps(splits, indent=1), encoding="utf-8", newline="\n")
    for split in ("train", "validation", "test"):
        total = sum(counts[name][split] for name in CATEGORIES)
        x = np.lib.format.open_memmap(out / (split + "-x.npy"), mode="w+", dtype=np.uint8, shape=(total, 28, 28))
        y = np.lib.format.open_memmap(out / (split + "-y.npy"), mode="w+", dtype=np.int64, shape=(total,))
        off = 0
        for ci, name in enumerate(CATEGORIES):
            a = np.load(out / f"{ci:03d}-{split}.npy", mmap_mode="r")
            x[off:off+len(a)] = a
            y[off:off+len(a)] = ci
            off += len(a)
        del x, y
    samples.sort(key=lambda s: s["label"])
    (out / "test_strokes.json").write_text(json.dumps(samples), encoding="utf-8", newline="\n")
    (out / "counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8", newline="\n")
    print("prepared", {split: sum(c[split] for c in counts.values()) for split in ("train", "validation", "test")}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-original", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--out", default="expanded", help="corpus folder under data/")
    ap.add_argument("--n-train", type=int, default=40000, help="max training drawings per category")
    ap.add_argument("--seed-splits", help="earlier corpus folder whose splits.json seeds a new corpus")
    args = ap.parse_args()
    out = DATA / args.out
    if args.pin_original:
        pin_original(out)
    if args.download:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(download, CATEGORIES):
                print(*result, flush=True)
    if args.prepare:
        prepare(out, args.n_train, args.seed_splits)
