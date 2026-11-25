"""Train-only streaming data for the unattended v4 round.

The append-only manifest is read to a complete newline. Prompt-hash splits are
checked before any sample is loaded. Validation is pinned once; test is ignored.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from categories import CATEGORIES
from colour import NET
from colour_batch import sample, sample_fills
from colour_data import ColourData, inputs
from fill import fill_planes
from generated_splits import assign, group_key
from generated_convert import VERSION
from style_augment import augment

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / 'data/generated'


def rows():
    with (GENERATED / 'manifest.jsonl').open(encoding='utf-8') as source:
        for line in source:
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            if row.get('keep') and row.get('conversion_version') == VERSION:
                assert row['split'] == assign(row)
                assert row.get('split_group', group_key(row)) == group_key(row)
                yield row


def load_rows(selected):
    images, labels, logits, ids = [], [], [], []
    for row in selected:
        path = GENERATED / row['sample']
        assert path.resolve().is_relative_to(GENERATED.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sample_sha256']
        with np.load(path) as sample:
            assert int(sample['y']) == int(row['label'])
            assert sample['x'].shape == (4, 40, 40)
            images.append(sample['x']); labels.append(int(sample['y']))
            logits.append(sample['teacher_logits'])
        ids.append(row['id'])
    return {'x': np.stack(images), 'y': np.asarray(labels, np.int64),
            'teacher': np.stack(logits).astype(np.float32), 'ids': np.asarray(ids)}


def pin_validation(folder):
    marker = folder / 'generated-validation.json'
    bank = folder / 'generated-validation.npz'
    if marker.exists():
        meta=json.loads(marker.read_text())
        if not bank.exists() or hashlib.sha256(bank.read_bytes()).hexdigest()!=meta['sha256']:
            # Rebuild only from the already pinned validation identities.
            temporary=bank.with_suffix('.repair.npz')
            np.savez_compressed(temporary,**load_rows(meta['rows']))
            assert hashlib.sha256(temporary.read_bytes()).hexdigest()==meta['sha256']
            temporary.replace(bank)
        return meta
    selected, seen = [], set()
    for row in rows():
        if row['split'] != 'validation' or row['input_sha256'] in seen:
            continue
        seen.add(row['input_sha256']); selected.append(row)
    assert selected, 'Need generated validation before unattended training'
    np.savez_compressed(bank, **load_rows(selected))
    meta = {'conversion': VERSION, 'rows': selected,
            'groups': sorted({group_key(r) for r in selected}),
            'sha256': hashlib.sha256(bank.read_bytes()).hexdigest()}
    marker.write_text(json.dumps(meta, indent=2), encoding='utf-8', newline='\n')
    return meta


class GrowingBank:
    def __init__(self, folder):
        self.validation = pin_validation(folder)
        self.data = None
        self.count = 0
        self.refresh()

    def refresh(self):
        selected, seen = [], set()
        forbidden = set(self.validation['groups'])
        for row in rows():
            if row['split'] != 'train' or row['input_sha256'] in seen:
                continue
            assert group_key(row) not in forbidden
            seen.add(row['input_sha256']); selected.append(row)
        # At the planned rate the bank remains < 10 GB over this 60h run.
        if len(selected) == self.count:
            return False
        self.data = load_rows(selected)
        self.count = len(selected)
        self.real = np.flatnonzero(self.data['y'] < 270)
        self.custom = np.flatnonzero(self.data['y'] >= 270)
        self.custom_by_class = [np.flatnonzero(self.data['y'] == i) for i in range(270,299)]
        self.groups = sorted({group_key(r) for r in selected})
        self.signature = hashlib.sha256('\n'.join(self.data['ids']).encode()).hexdigest()
        return True


class Data:
    def __init__(self, folder):
        self.corpus = ColourData('train', 'colour299-40')
        self.regions = np.load(self.corpus.root / 'train-regions.npy', mmap_mode='r')
        custom = np.flatnonzero(self.corpus.y >= 270)
        cy = np.asarray(self.corpus.y[custom])
        self.custom = [custom[cy == i] for i in range(270,299)]
        self.bank = GrowingBank(folder)
        with np.load(ROOT / 'data/synth-diverse40/train.npz') as z:
            self.proc = {k: z[k] for k in z.files}
        self.proc_by_class = [np.flatnonzero(self.proc['y'] == i) for i in range(270,299)]
        with np.load(ROOT / 'data/generated/rounds/balanced-v4-20261110/qd-game/train.npz') as z:
            self.game = {k: z[k] for k in ['x', 'y']}
        assert np.all(self.game['y'] < 270)

    def random_ids(self, count, custom, rng):
        if custom:
            return np.asarray([rng.choice(self.custom[c]) for c in rng.integers(0,29,count)])
        ids = rng.integers(0,len(self.corpus.y),count)
        bad = self.corpus.y[ids] >= 270
        while bad.any():
            ids[bad] = rng.integers(0,len(self.corpus.y),bad.sum())
            bad = self.corpus.y[ids] >= 270
        return ids

    def qd(self, ids, rng):
        d = self.corpus
        y = np.array(d.y[ids]); ink = torch.from_numpy(np.array(d.x[ids])).cuda()
        owner = torch.from_numpy(np.array(d.owner[ids],dtype=np.int32)).cuda()
        spans = [(int(d.offset[i]),int(d.offset[i+1])) for i in ids]
        lengths = np.concatenate([d.length[lo:hi] for lo,hi in spans])
        offset = np.r_[0,np.cumsum([hi-lo for lo,hi in spans])].astype(np.int64)
        seed = int(rng.integers(1,2**30))
        colours = sample(y,offset,lengths,seed,0)
        fills = sample_fills(y,offset,lengths,colours,seed+700000)
        chunk = (ink,torch.from_numpy(y).cuda(),owner,
                 torch.from_numpy(offset[:-1]).cuda(),torch.from_numpy(colours).cuda())
        x, target = inputs(chunk)
        region = torch.from_numpy(np.unpackbits(np.array(self.regions[ids]),axis=1,count=1600)
                                  .reshape(-1,40,40).astype(bool)).cuda()
        rgb = torch.as_tensor(NET,device='cuda')[torch.from_numpy(fills).cuda().long()]
        return styles(x,region,rgb,target,rng), target

    def batch(self, rng, size=64):
        # 50% real negatives, 50% custom. Generated custom never exceeds generated
        # real + converted-QD negatives. Original custom rehearsal keeps ALL styles.
        unit = size // 16
        pieces = [self.qd(self.random_ids(6*unit,False,rng),rng),
                  self.qd(self.random_ids(5*unit,True,rng),rng)]
        ix = np.asarray([rng.choice(self.proc_by_class[c]) for c in rng.integers(0,29,2*unit)])
        px = torch.from_numpy(self.proc['x'][ix].astype(np.float32)/255).cuda()
        py = torch.from_numpy(self.proc['y'][ix].astype(np.int64)).cuda()
        region = torch.from_numpy(self.proc['region'][ix]).cuda()
        rgb = torch.as_tensor(NET,device='cuda')[torch.from_numpy(self.proc['fill'][ix]).cuda().long()]
        pieces.append((styles(px,region,rgb,py,rng),py))
        # One converted-QD negative unit and one real + one custom generated unit.
        ix = rng.integers(0,len(self.game['y']),unit)
        pieces.append(bank_tensor(self.game,ix))
        kd_start = sum(len(y) for _,y in pieces)
        b = self.bank
        real_ids = rng.choice(b.real,unit)
        present = [v for v in b.custom_by_class if len(v)]
        custom_ids = np.asarray([rng.choice(present[int(c)]) for c in rng.integers(0,len(present),unit)])
        chosen = np.r_[real_ids,custom_ids]
        pieces.append(bank_tensor(b.data,chosen))
        x = torch.cat([p[0] for p in pieces]).contiguous(memory_format=torch.channels_last)
        y = torch.cat([p[1] for p in pieces])
        # Colour is optional, including generated paints. Keep geometry identical.
        black = torch.from_numpy(rng.random(len(x)) < .30).cuda()
        x[black,1:] = 0
        teacher = torch.from_numpy(b.data['teacher'][chosen]).cuda()
        return x,y,kd_start,teacher


def bank_tensor(bank, ix):
    return (torch.from_numpy(bank['x'][ix].astype(np.float32)/255).cuda(),
            torch.from_numpy(bank['y'][ix].astype(np.int64)).cuda())


def styles(x,region,rgb,y,rng):
    """Explicit black/colour/filled/brush rehearsal; do not fill text into blobs.

    Training-only overpainting is suppressed for detail-dependent custom classes.
    Canonical filled/brush validation deliberately remains unchanged.
    """
    original = x.clone()
    choice = torch.from_numpy(rng.random(len(x))).cuda()
    no_blob = torch.zeros(299,device='cuda',dtype=torch.bool)
    for name in ['spiral','x2','undo arrow','cursed face','robot','straw hat']:
        no_blob[CATEGORIES.index(name)] = True
    region = region & ~no_blob[y][:,None,None]
    filled = choice < .30
    fill_planes(x,rgb,filled,region=region)
    brush = choice >= .70
    if brush.any():
        bx = augment(original[brush],region[brush],rgb[brush],preserve_details=True)
        x[brush] = bx
    return x.contiguous(memory_format=torch.channels_last)
