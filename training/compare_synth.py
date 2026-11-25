"""Separate real-data retention, synthetic-test accuracy and false alarms.

Synthetic held-outs share drawing programs with training: their accuracy is an
optimistic upper bound, not evidence about real people's drawings.
"""
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def accuracy(top, truth):
    return {"drawings": len(truth), "top1": float(np.mean(top[:, 0] == truth)),
            "top3": float(np.mean(np.any(top == truth[:, None], axis=1)))}


def compare(baseline, candidate):
    with np.load(HERE / "out" / (baseline + ".predictions.npz")) as z:
        old_truth, old_top = z["truth"], z["top3"]
    with np.load(HERE / "out" / (candidate + ".predictions.npz")) as z:
        truth, top = z["truth"], z["top3"]
    meta = json.loads((HERE / "out" / (candidate + ".json")).read_text())
    labels = meta["categories"]
    real = truth < 270
    assert np.array_equal(truth[real], old_truth), "old test IDs/row order changed"
    audit = json.loads((HERE.parent / "data" / meta["corpus"] / "audit.json").read_text())
    assert audit["real_splits_identical"] and all(audit["real_heldout_pixels_identical"].values())
    before, after = accuracy(old_top, old_truth), accuracy(top[real], truth[real])
    synthetic = accuracy(top[~real], truth[~real])
    false = top[real, 0] >= 270
    report = {"baseline": baseline, "candidate": candidate, "size": meta["size"],
              "method": meta.get("training_mode", "full real-data replay fine-tuning with old-network distillation"),
              "real_before": before, "real_after": after,
              "real_delta_points": {k: 100*(after[k]-before[k]) for k in ("top1", "top3")},
              "synthetic_test": synthetic,
              "synthetic_caveat": "Upper bound only: held-outs share the same generator programs; real-player generalisation is unmeasured.",
              "real_to_synthetic": {"count": int(false.sum()), "denominator": int(real.sum()),
                                    "rate": float(false.mean()),
                                    "top3_any_synthetic_rate": float(np.any(top[real] >= 270, axis=1).mean())},
              "per_synthetic_class": {}}
    real_truth, real_top = truth[real], top[real]
    for ci, label in enumerate(labels[270:], 270):
        mask = truth == ci
        alarms = real_top[:, 0] == ci
        sources = np.bincount(real_truth[alarms], minlength=270)
        report["per_synthetic_class"][label] = {
            **accuracy(top[mask], truth[mask]), "real_false_alarms": int(alarms.sum()),
            "real_false_alarm_rate": float(alarms.mean()),
            "real_sources": [{"label": labels[j], "count": int(sources[j])}
                             for j in np.argsort(-sources)[:5] if sources[j]]}
    report["worst_synthetic_false_alarms"] = sorted(
        [{"label": n, **m} for n, m in report["per_synthetic_class"].items()],
        key=lambda m: -m["real_false_alarms"])
    report["per_real_class"] = {}
    for ci, label in enumerate(labels[:270]):
        mask = real_truth == ci
        b, a = accuracy(old_top[mask], old_truth[mask]), accuracy(real_top[mask], real_truth[mask])
        report["per_real_class"][label] = {"before": b, "after": a,
            "top1_delta_points": 100*(a["top1"]-b["top1"]),
            "synthetic_false_alarms": int((real_top[mask, 0] >= 270).sum())}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="wide270_28")
    ap.add_argument("--candidate", default="wide299_28")
    ap.add_argument("--out", default=str(HERE.parent / "notes/synthetic-comparison-299.json"))
    args = ap.parse_args()
    report = compare(args.baseline, args.candidate)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_real_class", "per_synthetic_class", "worst_synthetic_false_alarms")}, indent=1))
    print("Largest real-to-synthetic false alarms:")
    for row in report["worst_synthetic_false_alarms"][:8]:
        print(row["label"], row["real_false_alarms"], f"{100*row['real_false_alarm_rate']:.4f}% of all real test drawings")
