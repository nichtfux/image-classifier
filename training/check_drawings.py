"""Try original synthetic doodles against an exported model (not training data)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import raster
from train import dequant, forward_plain

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cases", default="data/smoke_strokes.json")
    ap.add_argument("--output", default="notes/handdrawn-check.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    torch.set_num_threads(4)
    meta = json.loads((root / "training/out" / (args.name + ".json")).read_text())
    with np.load(root / "training/out" / (args.name + ".npz")) as z:
        layers = dequant([(z[f"w{i}"], z[f"s{i}"], z[f"b{i}"]) for i in range(5)])
    renderer = raster.Renderer(meta.get("size", 28))
    cases = json.loads((root / args.cases).read_text())
    results = []
    for case in cases:
        drawing = [[list(p[0] for p in stroke), list(p[1] for p in stroke)] for stroke in case["strokes"]]
        image = torch.from_numpy(renderer.render(drawing))[None, None]
        with torch.no_grad():
            probs = forward_plain(layers, image).softmax(1)[0]
        values, indices = probs.topk(3)
        results.append({"intended": case["label"], "top": [
            {"label": meta["categories"][int(i)], "prob": float(p)} for i, p in zip(indices, values)]})
    (root / args.output).write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=1))
