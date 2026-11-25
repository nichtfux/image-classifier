"""Check append-only initialization, replay/distillation and false-alarm denominators."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
import train
import train_head
import compare_synth
from categories import CATEGORIES


if __name__ == "__main__":
    torch.set_num_threads(2)
    old = train.Net([3,6,9], 12, 270).eval()
    new = train.Net([3,6,9], 12, 299).eval()
    checkpoint = {"categories": CATEGORIES[:270], "state_dict": old.state_dict(), "args": {"epochs": 0}}
    assert train.extend_from_checkpoint(new, checkpoint, CATEGORIES) == 270
    probe = torch.rand(4,1,28,28)
    with torch.no_grad():
        assert torch.equal(old(probe), new(probe)[:,:270]), "old logits changed before training"
        extracted = train_head.features(old, probe)
        assert torch.equal(old(probe), torch.nn.functional.linear(extracted, old.f2.weight, old.f2.bias))
    assert torch.all(new.f2.bias[270:] == -3)
    branch = train_head.ExtraHead(old.f1.in_features, old.f1.out_features, 6)
    branched = train_head.assemble(checkpoint, CATEGORIES, branch, [3,6,9]).eval()
    with torch.no_grad():
        assert torch.allclose(old(probe), branched(probe)[:,:270], atol=1e-6, rtol=1e-6)
    oq, nq = train.quantise(train.fold(old)), train.quantise(train.fold(branched))
    for i, ((ow,os,ob),(nw,ns,nb)) in enumerate(zip(oq,nq)):
        if i==3: nw,ns,nb=nw[:12],ns[:12],nb[:12]
        if i==4: nw,ns,nb=nw[:270,:12],ns[:270],nb[:270]
        assert np.array_equal(ow,nw) and np.array_equal(os,ns) and np.array_equal(ob,nb)
    assert not nq[4][0][:270,12:].any()
    with tempfile.TemporaryDirectory(prefix="synth-smoke-", dir=Path(__file__).resolve().parent / "out") as folder:
        root = Path(folder)
        here, data = root / "training", root / "data"
        corpus, out = data / "mini", here / "out"
        corpus.mkdir(parents=True); out.mkdir(parents=True)
        rng = np.random.default_rng(8)
        for split, n in (("train", 598), ("validation", 299), ("test", 598)):
            np.save(corpus / (split+"-x.npy"), rng.integers(0,256,(n,28,28),dtype=np.uint8))
            np.save(corpus / (split+"-y.npy"), np.arange(n,dtype=np.int64)%299)
        (corpus/"splits.json").write_text(json.dumps({"categories":CATEGORIES}))
        (corpus/"corpus.json").write_text(json.dumps({"training_order":"interleaved256"}))
        (corpus/"audit.json").write_text(json.dumps({"real_splits_identical":True,"real_heldout_pixels_identical":{"test":True,"validation":True}}))
        torch.save(checkpoint, out/"old.pt")
        train.DATA, train.OUT = str(data), str(out)
        sys.argv = ["train.py", "--corpus", "mini", "--name", "candidate", "--widths", "3,6,9", "--fc", "12",
                    "--epochs", "1", "--chunk", "128", "--bs", "64", "--eval-bs", "128", "--clean",
                    "--init", str(out/"old.pt"), "--max-real-drop", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            train.main()
        assert np.load(out/"candidate.npz")["w4"].shape == (299,12)
        sys.argv = ["train.py", "--corpus", "mini", "--name", "candidate_eval", "--widths", "3,6,9", "--fc", "12",
                    "--epochs", "0", "--resume", str(out/"candidate.pt"), "--clean", "--eval-bs", "128"]
        with contextlib.redirect_stdout(io.StringIO()):
            train.main()
        with np.load(out/"candidate.npz") as first, np.load(out/"candidate_eval.npz") as evaluated:
            assert all(np.array_equal(first[k], evaluated[k]) for k in first.files), "evaluation changed model weights"
        # Controlled outcomes: exactly two real false alarms, both new labels.
        old_truth = np.repeat(np.arange(270,dtype=np.int16),2)
        new_truth = np.repeat(np.arange(270,299,dtype=np.int16),2)
        truth = np.r_[old_truth,new_truth]
        old_top = np.stack([old_truth,(old_truth+1)%270,(old_truth+2)%270],1)
        top = np.stack([truth,(truth+1)%299,(truth+2)%299],1)
        top[0] = [270,0,1]; top[1] = [290,0,1]
        np.savez(out/"old.predictions.npz",truth=old_truth,top3=old_top)
        np.savez(out/"candidate.predictions.npz",truth=truth,top3=top)
        compare_synth.HERE = here
        report = compare_synth.compare("old","candidate")
        assert report["real_to_synthetic"]["count"] == 2
        assert report["real_to_synthetic"]["denominator"] == 540
        assert report["real_to_synthetic"]["rate"] == 2/540
        assert report["per_synthetic_class"]["bomb"]["real_false_alarms"] == 1
        assert report["per_synthetic_class"]["heart"]["real_false_alarms"] == 1
        assert report["synthetic_test"]["top1"] == 1
        assert report["real_after"]["top1"] == 538/540
    print("Append-only initialization, small replay/distillation run and false-alarm fixture passed.")
