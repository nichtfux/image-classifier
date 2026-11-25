"""Train the small CNN, fold batch-norm, quantise weights to int8 (per output channel)
and report test accuracy of the shipped (quantised) model.

    # original 32-category run (data/rendered.npz from prepare.py)
    python train.py --name small --widths 16,32,64 --fc 128 --epochs 12
    # a corpus built by expand_data.py (data/<corpus>/)
    python train.py --corpus expanded100 --name wide100 --widths 24,48,96 --fc 256 --epochs 6
    # optional: a big teacher network (GPU only, never exported) and distillation from it
    python train.py --corpus expanded100 --arch teacher --name teacher100 --epochs 3
    python train.py --corpus expanded100 --name wide100kd --widths 24,48,96 --fc 256 --teacher out/teacher100.pt

Writes out/<name>.npz (folded int8 weights + scales + float biases), out/<name>.json
(results) and out/<name>.pt (checkpoint). Labels are the corpus's own category list,
which is always a prefix of categories.py (append-only).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from categories import CATEGORIES

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "out")


class Net(nn.Module):
    """The shipped network: 3x (conv 3x3, BN, ReLU, max-pool 2), dense, dense."""

    def __init__(self, widths, fc, ncls, size=28, in_channels=1):
        super().__init__()
        c1, c2, c3 = widths
        self.c1 = nn.Conv2d(in_channels, c1, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(c1)
        self.c2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(c2)
        self.c3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False); self.b3 = nn.BatchNorm2d(c3)
        self.f1 = nn.Linear(c3 * (size // 8) ** 2, fc)
        self.f2 = nn.Linear(fc, ncls)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)  # 14
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)  # 7
        x = F.max_pool2d(F.relu(self.b3(self.c3(x))), 2)  # 3
        x = self.drop(F.relu(self.f1(x.flatten(1))))
        return self.f2(x)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.a = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.ba = nn.BatchNorm2d(cout)
        self.b = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.bb = nn.BatchNorm2d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.ba(self.a(x)))
        y = self.bb(self.b(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class Teacher(nn.Module):
    """A ResNet-style network for distillation only (far too slow for Luau)."""

    def __init__(self, ncls, w=(64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        self.s1 = nn.Sequential(Block(w[0], w[0], 1), Block(w[0], w[0], 1))   # 28
        self.s2 = nn.Sequential(Block(w[0], w[1], 2), Block(w[1], w[1], 1))   # 14
        self.s3 = nn.Sequential(Block(w[1], w[2], 2), Block(w[2], w[2], 1))   # 7
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(w[2], ncls))

    def forward(self, x):
        x = self.s3(self.s2(self.s1(self.stem(x))))
        return self.head(F.adaptive_avg_pool2d(x, 1))


def fold(net):
    """Return list of (W, b) float64 numpy for conv1..3, fc1, fc2 with BN folded in."""
    layers = []
    for conv, bn in ((net.c1, net.b1), (net.c2, net.b2), (net.c3, net.b3)):
        g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
        w = conv.weight.detach().double() * g[:, None, None, None]
        b = bn.bias.detach().double() - bn.running_mean.detach().double() * g
        layers.append((w.cpu().numpy(), b.cpu().numpy()))
    for lin in (net.f1, net.f2):
        layers.append((lin.weight.detach().double().cpu().numpy(), lin.bias.detach().double().cpu().numpy()))
    return layers


def quantise(layers):
    """int8 per output channel. Biases are rounded to float32 (what Luau stores)."""
    q = []
    for w, b in layers:
        flat = w.reshape(w.shape[0], -1)
        scale = np.abs(flat).max(1) / 127.0
        scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
        wi = np.clip(np.round(flat / scale[:, None].astype(np.float64)), -127, 127).astype(np.int8)
        q.append((wi.reshape(w.shape), scale, b.astype(np.float32)))
    return q


def dequant(q):
    return [(wi.astype(np.float64) * s.astype(np.float64).reshape(
        (wi.shape[0],wi.shape[1],1,1) if s.ndim==2 else ((-1,)+(1,)*(wi.ndim-1))), b.astype(np.float64)) for wi,s,b in q]


def forward_plain(layers, x):
    """Reference forward pass with plain (folded/dequantised) weights, float64. x: (N,1,28,28)."""
    (w1, b1), (w2, b2), (w3, b3), (f1, g1), (f2, g2) = [(torch.from_numpy(w), torch.from_numpy(b)) for w, b in layers]
    x = F.max_pool2d(F.relu(F.conv2d(x, w1, b1, padding=1)), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, w2, b2, padding=1)), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, w3, b3, padding=1)), 2)
    x = F.relu(F.linear(x.flatten(1), f1, g1))
    return F.linear(x, f2, g2)


def topk_acc(logits, y):
    top = logits.topk(3, 1).indices
    return (top[:, 0] == y).float().mean().item(), (top == y[:, None]).any(1).float().mean().item()


def extend_from_checkpoint(net, checkpoint, categories):
    """Copy the old body and output rows exactly; append only new output rows."""
    old_labels = checkpoint["categories"]
    assert old_labels == categories[:len(old_labels)] and len(old_labels) < len(categories)
    state = checkpoint["state_dict"]
    current = net.state_dict()
    for key, value in state.items():
        if key not in ("f2.weight", "f2.bias"):
            assert current[key].shape == value.shape, key
            current[key].copy_(value)
    n = len(old_labels)
    current["f2.weight"][:n].copy_(state["f2.weight"])
    current["f2.bias"][:n].copy_(state["f2.bias"])
    # Initially avoid stealing old predictions with random new-class logits.
    current["f2.weight"][n:].normal_(0, 0.01)
    current["f2.bias"][n:].fill_(-3.0)
    net.load_state_dict(current)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="small")
    ap.add_argument("--arch", choices=("small", "teacher"), default="small")
    ap.add_argument("--widths", default="16,32,64")
    ap.add_argument("--fc", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--corpus", help="corpus folder under data/ built by expand_data.py")
    ap.add_argument("--expanded", action="store_true", help="same as --corpus expanded (the 64-category round)")
    ap.add_argument("--resume", help="Continue from an existing training checkpoint")
    ap.add_argument("--init", help="Append output classes to this small-network checkpoint")
    ap.add_argument("--preserve-weight", type=float, default=0.5, help="old-class distillation strength with --init")
    ap.add_argument("--max-real-drop", type=float, default=0.005, help="validation selection guard for original classes")
    ap.add_argument("--max-real-false-alarm", type=float, default=0.01, help="validation real-to-new top-1 limit")
    ap.add_argument("--clean", action="store_true", help="Refine on the exact inference raster without geometric augmentation")
    ap.add_argument("--augment-prob", type=float, default=0.8)
    ap.add_argument("--teacher", help="checkpoint of a --arch teacher run to distil from")
    ap.add_argument("--kd-alpha", type=float, default=0.7, help="weight of the distillation loss")
    ap.add_argument("--kd-temp", type=float, default=3.0)
    ap.add_argument("--chunk", type=int, default=1_000_000, help="training images moved to the GPU at a time")
    ap.add_argument("--size", type=int, help="input resolution (inferred from corpus, checked if supplied)")
    ap.add_argument("--eval-bs", type=int, default=1024, help="bounded evaluation batch size")
    a = ap.parse_args()
    if a.expanded and not a.corpus:
        a.corpus = "expanded"
    widths = [int(v) for v in a.widths.split(",")]
    torch.manual_seed(0)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

    if a.corpus:
        corpus = os.path.join(DATA, a.corpus)
        split_info = json.load(open(os.path.join(corpus, "splits.json")))
        categories = split_info["categories"]
        assert categories == CATEGORIES[:len(categories)], "corpus labels are not a prefix of categories.py"
        def read(split, xy):
            # Windows reserves commit for the full file with copy-on-write maps.
            # Read-only mappings reserve no 40GB private copy of the corpus.
            return np.load(os.path.join(corpus, split + "-" + xy + ".npy"), mmap_mode="r")
        # Keep training on disk: 270 classes at 40px can exceed 40 GB. Only a
        # bounded chunk and its prefetched successor are copied into RAM/GPU.
        xtr, ytr = read("train", "x"), read("train", "y")
        xv, yv = torch.from_numpy(np.array(read("validation", "x"))).to(dev), torch.from_numpy(np.array(read("validation", "y"))).to(dev)
        xte, yte = torch.from_numpy(np.array(read("test", "x"))).to(dev), torch.from_numpy(np.array(read("test", "y"))).to(dev)
        meta_path = os.path.join(corpus, "corpus.json")
        corpus_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    else:
        categories = CATEGORIES[:32]
        d = np.load(os.path.join(DATA, "rendered.npz"))
        xtr = torch.from_numpy(d["xtr"])
        ytr = torch.from_numpy(d["ytr"])
        xte = torch.from_numpy(d["xte"]).to(dev)
        yte = torch.from_numpy(d["yte"]).to(dev)
        xv, yv = xte, yte
        assert int(ytr.max()) + 1 == len(categories), "stale rendered.npz labels: rerun prepare.py"
        corpus_meta = {}
    size = int(xtr.shape[-1])
    assert xtr.shape[1:] == (size, size)
    assert a.size is None or a.size == size, "input size differs from corpus"
    a.size = size
    ncls = len(categories)
    augment = a.corpus is not None and not a.clean
    print("classes", ncls, "train", len(xtr), "validation", len(xv), "test", len(xte), flush=True)

    if a.arch == "teacher":
        net = Teacher(ncls).to(dev)
    else:
        net = Net(widths, a.fc, ncls, size=size).to(dev)
    net = net.to(memory_format=torch.channels_last)
    preserver, real_classes = None, 0
    assert not (a.init and a.resume), "choose --init or --resume"
    if a.init:
        assert a.arch == "small"
        initial = torch.load(a.init, map_location=dev, weights_only=True)
        real_classes = extend_from_checkpoint(net, initial, categories)
        preserver = Net(widths, a.fc, real_classes, size=size).to(dev).to(memory_format=torch.channels_last)
        preserver.load_state_dict(initial["state_dict"])
        preserver.eval().requires_grad_(False)
        print("fine-tuning", a.init, "old classes", real_classes, "new classes", ncls-real_classes, flush=True)
    teacher = None
    if a.teacher:
        tc = torch.load(a.teacher, map_location=dev, weights_only=True)
        assert tc["categories"] == categories, "teacher labels differ"
        teacher = Teacher(ncls).to(dev).to(memory_format=torch.channels_last)
        teacher.load_state_dict(tc["state_dict"])
        teacher.eval()
    prior_epochs, prior_seconds = 0, 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location=dev, weights_only=True)
        assert checkpoint["categories"] == categories, "checkpoint labels changed"
        net.load_state_dict(checkpoint["state_dict"])
        prior_epochs = checkpoint.get("total_epochs", checkpoint["args"]["epochs"])
        meta_path = os.path.splitext(a.resume)[0] + ".json"
        if os.path.exists(meta_path):
            prior_seconds = json.load(open(meta_path)).get("train_seconds", 0)
    nparams = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * sum((min(a.chunk, len(xtr) - i) + a.bs - 1) // a.bs for i in range(0, len(xtr), a.chunk))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps) if steps else None
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    def to_input(xb):
        return xb.float().div_(255).unsqueeze(1).contiguous(memory_format=torch.channels_last)

    def evaluate(model, x, cpu=False, mixed=True):
        # Preallocate once instead of retaining every batch and concatenating a
        # second full 540k-by-270 logit matrix on the GPU.
        output_classes = model.f2.out_features if isinstance(model, Net) else ncls
        result = torch.empty((len(x), output_classes), dtype=torch.float32, device="cpu" if cpu else dev)
        with torch.no_grad(), torch.autocast(**{**amp,"enabled":amp["enabled"] and mixed}):
            for i in range(0, len(x), a.eval_bs):
                logits = model(to_input(x[i:i+a.eval_bs])).float()
                result[i:i+len(logits)] = logits.cpu() if cpu else logits
        return result

    def augment_images(x):
        # Small rotations, independent stretch/shear and one-pixel translations.
        # Unlike a fixed second image, these change every time a drawing is seen.
        n = len(x)
        angle = torch.empty(n, device=dev).uniform_(-0.25, 0.25)
        sx = torch.empty(n, device=dev).uniform_(0.85, 1.15)
        sy = torch.empty(n, device=dev).uniform_(0.85, 1.15)
        shear = torch.empty(n, device=dev).uniform_(-0.12, 0.12)
        theta = torch.zeros(n, 2, 3, device=dev)
        c, s = angle.cos(), angle.sin()
        theta[:, 0, 0], theta[:, 0, 1] = c * sx, c * shear - s * sy
        theta[:, 1, 0], theta[:, 1, 1] = s * sx, s * shear + c * sy
        theta[:, :, 2].uniform_(-0.07, 0.07)
        # Keep some clean images in every batch so validation matches training.
        clean = torch.rand(n, device=dev) >= a.augment_prob
        theta[clean] = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=dev)
        return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)

    def gather(idx):
        # New paired corpora are interleaved by class, permitting sequential disk
        # reads. Legacy class-contiguous corpora retain globally shuffled gathers.
        if isinstance(idx, tuple):
            start, end = idx
            xb, yb = np.array(xtr[start:end]), np.array(ytr[start:end])
        elif isinstance(xtr, np.ndarray):
            indices = idx.sort().values.numpy()
            xb, yb = xtr[indices], ytr[indices]
        else:
            indices = idx.sort().values
            return xtr[indices].to(dev), ytr[indices].to(dev)
        return torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)

    loader = ThreadPoolExecutor(1)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    last_progress = t0
    best_score, best_state, best_epoch = -1, None, 0
    # Keep the paired corpora's chunk order independent of the different-sized
    # dense layer initialization, which consumes a different amount of CPU RNG.
    data_rng = torch.Generator().manual_seed(0)
    real_val_mask = yv < real_classes
    real_validation_floor, real_validation_top3_floor = None, None
    if preserver is not None:
        baseline_val, baseline_val3 = topk_acc(evaluate(preserver, xv[real_val_mask]), yv[real_val_mask])
        real_validation_floor = baseline_val - a.max_real_drop
        real_validation_top3_floor = baseline_val3 - a.max_real_drop
        print(f"old validation baseline {baseline_val:.6f}/{baseline_val3:.6f}; "
              f"selection floor {real_validation_floor:.6f}/{real_validation_top3_floor:.6f}", flush=True)
    if a.resume:
        net.eval()
        best_score = topk_acc(evaluate(net, xv), yv)[0]
        best_state = copy.deepcopy(net.state_dict())
        print(f"starting validation top1 {best_score:.4f}", flush=True)
    for ep in range(a.epochs):
        epoch_start = time.time()
        net.train()
        if corpus_meta.get("training_order") == "interleaved256":
            ranges = [(start, min(start+a.chunk, len(xtr))) for start in range(0, len(xtr), a.chunk)]
            chunks = [ranges[i] for i in torch.randperm(len(ranges), generator=data_rng).tolist()]
        else:
            chunks = list(torch.randperm(len(xtr)).split(a.chunk))
        pending = loader.submit(gather, chunks[0])
        tot, seen = 0.0, 0
        for ci in range(len(chunks)):
            xc, yc = pending.result()
            if ci + 1 < len(chunks):
                pending = loader.submit(gather, chunks[ci + 1])
            order = torch.randperm(len(xc), device=dev)
            for i in range(0, len(xc), a.bs):
                j = order[i:i + a.bs]
                xb, yb = to_input(xc[j]), yc[j]
                if augment:
                    xb = augment_images(xb)
                with torch.autocast(**amp):
                    logits = net(xb)
                    loss = F.cross_entropy(logits.float(), yb, label_smoothing=0.05)
                    if preserver is not None and a.preserve_weight:
                        real_mask = yb < real_classes
                        if real_mask.any():
                            with torch.no_grad():
                                # Keep the teacher batch shape fixed. Masking the
                                # input makes cuDNN benchmark every varying count
                                # of real examples in a batch, costing minutes.
                                old_logits = preserver(xb).float()[real_mask]
                            T = 2.0
                            keep_loss = F.kl_div(F.log_softmax(logits[real_mask, :real_classes].float()/T, 1),
                                                F.softmax(old_logits/T, 1), reduction="batchmean") * T*T
                            loss = loss + a.preserve_weight * keep_loss
                    if teacher is not None:
                        with torch.no_grad():
                            tl = teacher(xb).float()
                        T = a.kd_temp
                        kd = F.kl_div(F.log_softmax(logits.float() / T, 1), F.softmax(tl / T, 1), reduction="batchmean") * T * T
                        loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                tot += loss.item() * len(j)
                seen += len(j)
            del xc, yc, order, xb, yb, logits, loss
            if time.time() - last_progress >= 45 or ci + 1 == len(chunks):
                print(f"epoch {ep+1}/{a.epochs} chunk {ci+1}/{len(chunks)} train {seen/len(xtr):.1%} "
                      f"{seen/max(1,time.time()-epoch_start):,.0f} images/s", flush=True)
                last_progress = time.time()
        pending = None
        net.eval()
        validation_logits = evaluate(net, xv)
        t1, t3 = topk_acc(validation_logits, yv)
        acceptable = True
        if preserver is not None:
            real_v1, real_v3 = topk_acc(validation_logits[real_val_mask], yv[real_val_mask])
            synth_v1, synth_v3 = topk_acc(validation_logits[~real_val_mask], yv[~real_val_mask])
            val_false_alarm = (validation_logits[real_val_mask].argmax(1) >= real_classes).float().mean().item()
            acceptable = (real_v1 >= real_validation_floor and real_v3 >= real_validation_top3_floor
                          and val_false_alarm <= a.max_real_false_alarm)
            print(f"validation real {real_v1:.6f}/{real_v3:.6f} synthetic {synth_v1:.6f}/{synth_v3:.6f} "
                  f"real-to-synth {val_false_alarm:.6f} retention_guard={acceptable}", flush=True)
        del validation_logits
        if t1 > best_score and acceptable:
            best_score, best_epoch = t1, ep + 1
            best_state = copy.deepcopy(net.state_dict())
            checkpoint_path = os.path.join(OUT, a.name + ".pt")
            torch.save({"state_dict": {k:v.detach().cpu() for k,v in best_state.items()},
                        "categories": categories, "args": vars(a), "total_epochs": prior_epochs+ep+1,
                        "best_epoch": best_epoch, "validation_top1": best_score}, checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)
        print(f"epoch {ep+1:2d} loss {tot/seen:.4f} validation top1 {t1:.4f} top3 {t3:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    loader.shutdown(wait=True)
    assert best_state is not None, "No epoch met the old-class validation retention guard; no export produced"
    net.load_state_dict(best_state)
    net.eval()
    os.makedirs(OUT, exist_ok=True)
    yc = yte.cpu()
    res = {"name": a.name, "arch": a.arch, "epochs": a.epochs, "params": nparams, "categories": categories,
           "n_train": len(xtr), "n_validation": len(xv), "n_test": len(xte), "corpus": a.corpus,
           "best_epoch": best_epoch, "validation_top1": best_score, "teacher": a.teacher,
           "total_epochs": prior_epochs + a.epochs, "selected_epoch": prior_epochs + best_epoch,
           "clean_refinement": a.clean, "augment_probability": a.augment_prob if augment else 0,
           "size": size, "box": 22.0*size/28, "radius": size/28,
           "train_ids_sha256": corpus_meta.get("train_ids_sha256"), "corpus_metadata": corpus_meta}
    res.update({"initialized_from": a.init, "preserve_weight": a.preserve_weight if a.init else 0,
                "real_validation_floor": real_validation_floor,
                "real_validation_top3_floor": real_validation_top3_floor,
                "real_false_alarm_validation_limit": a.max_real_false_alarm if a.init else None})
    res["data_shuffle_seed"] = 0 if corpus_meta.get("training_order") == "interleaved256" else None
    if a.arch == "teacher":
        tl = evaluate(net, xte, cpu=True)
        res["float_top1"], res["float_top3"] = topk_acc(tl, yc)
        res["train_seconds"] = time.time() - t0 + prior_seconds
        print(json.dumps({k: v for k, v in res.items() if k != "categories"}, indent=1))
        torch.save({"state_dict": net.cpu().state_dict(), "categories": categories, "args": vars(a),
                    "total_epochs": prior_epochs + a.epochs}, os.path.join(OUT, a.name + ".pt"))
        with open(os.path.join(OUT, a.name + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        return

    q = quantise(fold(net))
    # Evaluate the actual exported int8/float32 values on GPU in float32 (no autocast);
    # the exporter separately produces float64 reference vectors for Luau parity.
    qnet = copy.deepcopy(net).float()
    plain = dequant(q)
    with torch.no_grad():
        for target, (w, b) in zip((qnet.c1, qnet.c2, qnet.c3), plain[:3]):
            target.weight.copy_(torch.from_numpy(w).to(dev))
        # Folded convolution biases need actual bias parameters; remove BN.
        for i in range(1, 4):
            conv = getattr(qnet, "c" + str(i))
            conv.bias = nn.Parameter(torch.from_numpy(plain[i-1][1]).float().to(dev))
            setattr(qnet, "b" + str(i), nn.Identity())
        for target, (w, b) in zip((qnet.f1, qnet.f2), plain[3:]):
            target.weight.copy_(torch.from_numpy(w).to(dev))
            target.bias.copy_(torch.from_numpy(b).to(dev))
    with torch.no_grad():
        fl = evaluate(net, xte, cpu=True)
        ql = evaluate(qnet, xte, cpu=True, mixed=False)
    f1, f3 = topk_acc(fl, yc)
    q1, q3 = topk_acc(ql, yc)
    np.savez_compressed(os.path.join(OUT, a.name + ".predictions.npz"),
                        truth=yc.numpy().astype(np.int16), top3=ql.topk(3,1).indices.numpy().astype(np.int16))
    pred = ql.argmax(1)
    per_class = [(pred[yc == c] == c).float().mean().item() for c in range(ncls)]
    macs = size**2 * widths[0] * 9 + (size//2)**2 * widths[1] * widths[0] * 9 + (size//4)**2 * widths[2] * widths[1] * 9 \
        + widths[2] * (size//8)**2 * a.fc + a.fc * ncls
    res.update({
        "widths": widths, "fc": a.fc, "macs": macs,
        "float_top1": f1, "float_top3": f3, "int8_top1": q1, "int8_top3": q3,
        "per_class_top1": dict(zip(categories, per_class)), "train_seconds": time.time() - t0 + prior_seconds,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if dev == "cuda" else 0,
    })
    confusion = torch.bincount(yc * ncls + pred, minlength=ncls**2).reshape(ncls, -1)
    res["confusions"] = {
        name: [{"label": categories[j], "count": int(confusion[i, j])}
               for j in torch.argsort(confusion[i], descending=True).tolist() if j != i][:3]
        for i, name in enumerate(categories)
    }
    # Accuracy on each earlier round's categories (same pinned test drawings).
    for n in (32, 64, 100, 270):
        if ncls > n:
            m = yc < n
            res[f"first{n}_top1"], res[f"first{n}_top3"] = topk_acc(ql[m], yc[m])
    group_path = os.path.join(HERE, "category_groups.json")
    if os.path.exists(group_path):
        groups = json.load(open(group_path))
        res["per_group"] = {}
        for name, labels in groups.items():
            indices = [categories.index(label) for label in labels if label in categories]
            if indices:
                mask = torch.isin(yc, torch.tensor(indices))
                g1, g3 = topk_acc(ql[mask], yc[mask])
                res["per_group"][name] = {"classes": len(indices), "drawings": int(mask.sum()), "top1": g1, "top3": g3}
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_class_top1", "categories", "confusions")}, indent=1))
    torch.save({"state_dict": net.cpu().state_dict(), "categories": categories, "args": vars(a),
                "total_epochs": prior_epochs + a.epochs}, os.path.join(OUT, a.name + ".pt"))
    arrs = {}
    for i, (wi, s, b) in enumerate(q):
        arrs[f"w{i}"], arrs[f"s{i}"], arrs[f"b{i}"] = wi, s, b
    np.savez(os.path.join(OUT, a.name + ".npz"), **arrs)
    with open(os.path.join(OUT, a.name + ".json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
