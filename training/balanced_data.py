"""Freeze v4 samples and prepare style-matched Quick, Draw! negatives."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from categories import CATEGORIES
from colour import NET, colourise, planes
from generated_convert import VERSION, convert, preview
from generated_splits import assign, group_key


PROJECT = Path(__file__).resolve().parent.parent
GENERATED = PROJECT / 'data/generated'
WORKER_DATA = {}


def write_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def complete_rows():
    with (GENERATED / 'manifest.jsonl').open(encoding='utf-8') as source:
        for line in source:
            if not line.endswith('\n'):
                break
            yield json.loads(line)


def freeze_v4(destination):
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'manifest.json').exists():
        for split in ['train', 'validation', 'test']:
            if not (destination / f'{split}.npz').exists():
                raise RuntimeError(f'Incomplete frozen bank at {destination}')
        print('Reusing immutable generated bank', destination, flush=True)
        return
    rows = [row for row in complete_rows()
            if row['keep'] and row['conversion_version'] == VERSION]
    unique = set()
    selected = []
    for row in rows:
        signature = row['input_sha256']
        if signature in unique:
            continue
        unique.add(signature)
        assert row['split'] == assign(row)
        selected.append(row)

    groups = {}
    counts = {}
    for split in ['train', 'validation', 'test']:
        cohort = [row for row in selected if row['split'] == split]
        images = []
        labels = []
        identities = []
        for row in cohort:
            path = GENERATED / row['sample']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sample_sha256']
            with np.load(path) as sample:
                images.append(sample['x'])
                labels.append(int(sample['y']))
            identities.append(row['id'])
        np.savez_compressed(destination / f'{split}.npz', x=np.stack(images),
                            y=np.asarray(labels, np.int64), ids=np.asarray(identities))
        groups[split] = sorted({group_key(row) for row in cohort})
        counts[split] = dict(Counter('real' if label < 270 else 'custom' for label in labels))
    assert not (set(groups['train']) & set(groups['validation']))
    assert not (set(groups['train']) & set(groups['test']))
    assert not (set(groups['validation']) & set(groups['test']))
    write_json(destination / 'manifest.json', {
        'conversion_version': VERSION,
        'source_rows': selected,
        'groups': groups,
        'counts': counts,
        'sha256': hashlib.sha256('\n'.join(row['id'] for row in selected).encode()).hexdigest(),
    })
    print('Frozen generated bank', len(selected), counts, flush=True)


def initialize_worker():
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    folder = PROJECT / 'data/colour299-40'
    metadata = json.loads((folder / 'corpus.json').read_text())
    base = PROJECT / 'data' / metadata['base']
    for split in ['train', 'validation']:
        WORKER_DATA[split] = {
            'x': np.load(base / f'{split}-x.npy', mmap_mode='r'),
            'owner': np.load(folder / f'{split}-owner.npy', mmap_mode='r'),
            'offset': np.load(folder / f'{split}-offset.npy', mmap_mode='r'),
            'length': np.load(folder / f'{split}-length.npy', mmap_mode='r'),
            'region': np.load(folder / f'{split}-regions.npy', mmap_mode='r'),
        }


def transform_negative(job):
    split, index, label, seed, role = job
    data = WORKER_DATA[split]
    rng = np.random.default_rng(seed)
    begin, end = int(data['offset'][index]), int(data['offset'][index + 1])
    lengths = np.asarray(data['length'][begin:end])
    colours = colourise(CATEGORIES[label], lengths, rng)
    if rng.random() < .2:
        colours = rng.integers(0, len(NET), len(colours)).tolist()
    ink = np.asarray(data['x'][index], np.float32) / 255.
    image = planes(ink, data['owner'][index], colours)
    style = rng.choice(['outline', 'flat', 'painted', 'kids'], p=[.25, .25, .35, .15])
    if style != 'outline':
        region = np.unpackbits(data['region'][index], count=1600).reshape(40, 40).astype(bool)
        fill_colour = int(rng.choice(colours))
        if style == 'painted' and rng.random() < .5:
            yy, xx = np.mgrid[:40, :40]
            angle = rng.uniform(0, 2 * np.pi)
            spacing = rng.uniform(2, 7)
            stripes = (xx * np.cos(angle) + yy * np.sin(angle)) % spacing
            region &= stripes < spacing * rng.uniform(.4, .9)
        image[0, region] = 1.
        image[1:, region] = NET[fill_colour, :, None]
    source = preview(np.rint(image * 255).astype(np.uint8))
    try:
        converted = convert(source, str(style), seed + 700)
    except ValueError:
        return None
    return converted['x'], label, index, role, str(style)


def prepare_qd_negatives(destination, train_per_class, validation_per_class, workers):
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'manifest.json'
    if marker.exists():
        metadata = json.loads(marker.read_text())
        assert metadata['requested'] == {'train_per_real_class': train_per_class,
                                         'validation_per_real_class': validation_per_class}
        assert all((destination / f'{split}.npz').exists() for split in ['train', 'validation'])
        print('Reusing immutable real-negative bank', destination, flush=True)
        return
    folder = PROJECT / 'data/colour299-40'
    metadata = json.loads((folder / 'corpus.json').read_text())
    base = PROJECT / 'data' / metadata['base']
    rng = np.random.default_rng(2026111101)
    jobs = []
    for split, count in [('train', train_per_class), ('validation', validation_per_class)]:
        labels = np.load(base / f'{split}-y.npy', mmap_mode='r')
        limit = min(len(labels), 500000)
        for label in range(270):
            candidates = np.flatnonzero(labels[:limit] == label)
            if split == 'train':
                chosen = rng.choice(candidates, count, replace=False)
                roles = np.zeros(count, np.uint8)
            else:
                assert len(candidates) == 1000
                half = count // 2
                chosen = np.concatenate([
                    rng.choice(candidates[:500], half, replace=False),
                    rng.choice(candidates[500:], count - half, replace=False),
                ])
                roles = np.r_[np.zeros(half, np.uint8), np.ones(count - half, np.uint8)]
            for index, role in zip(chosen, roles):
                jobs.append((split, int(index), label, int(rng.integers(1, 2**31)), int(role)))

    started = time.time()
    outputs = {'train': [], 'validation': []}
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        for number, (job, result) in enumerate(zip(jobs, pool.map(transform_negative, jobs, chunksize=16)), 1):
            if result is not None:
                outputs[job[0]].append(result)
            if number % 2000 == 0:
                print('Converted QD negatives', number, '/', len(jobs), 'seconds', round(time.time() - started), flush=True)
    for split, rows in outputs.items():
        images, labels, indices, roles, styles = zip(*rows)
        np.savez_compressed(destination / f'{split}.npz',
                            x=np.stack(images), y=np.asarray(labels, np.int64),
                            source_indices=np.asarray(indices, np.int64), role=np.asarray(roles, np.uint8),
                            styles=np.asarray(styles))
    write_json(destination / 'manifest.json', {
        'source': 'Recognized Quick, Draw! train and validation only',
        'conversion_version': VERSION,
        'counts': {split: len(rows) for split, rows in outputs.items()},
        'requested': {'train_per_real_class': train_per_class,
                      'validation_per_real_class': validation_per_class},
        'seconds': time.time() - started,
    })
    print('Prepared real-class negatives', {key: len(value) for key, value in outputs.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--train-per-class', type=int, default=80)
    parser.add_argument('--validation-per-class', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    arguments = parser.parse_args()
    destination = Path(arguments.out).resolve()
    freeze_v4(destination / 'generated')
    prepare_qd_negatives(destination / 'qd-game', arguments.train_per_class,
                         arguments.validation_per_class, arguments.workers)


if __name__ == '__main__':
    main()
