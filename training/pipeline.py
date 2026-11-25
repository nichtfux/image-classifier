"""Fine-tune at 28px using preserved real data and reviewed synthetic strokes.

Requires expanded270-28 and wide270_28.pt. Preserves the 40px artifacts.
Generation precedes corpus preparation; --skip-synth reuses reviewed files.
Never installs to Studio, saves or publishes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synth", action="store_true", help="reuse reviewed data/synth/*.ndjson")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=250000)
    ap.add_argument("--extra-hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--name", default="wide299_28_r4")
    ap.add_argument("--report", help="comparison report path; revision names get separate reports by default")
    args = ap.parse_args()
    if args.skip_prepare and not args.skip_synth:
        ap.error("--skip-prepare requires --skip-synth; regenerated drawings must be prepared again")
    here = Path(__file__).resolve().parent
    out, root = here / "out", here.parent
    out.mkdir(exist_ok=True)
    assert (root / "data/expanded270-28/splits.json").exists(), "Recover the pinned expanded270-28 corpus first"
    assert (out / "wide270_28.pt").exists(), "Recover the wide270_28 checkpoint first"
    completed = []

    def status(value):
        (out / "round299-status.json").write_text(json.dumps(value, indent=1), encoding="utf-8", newline="\n")

    def run(log_name, script, *options):
        command = [sys.executable, "-u", str(here / script), *options]
        with (out / log_name).open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(command, cwd=here, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            current = {"phase": log_name, "pid": process.pid, "command": command,
                       "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "completed": completed}
            status(current)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            code = process.wait()
        if code:
            current["exit_code"] = code; status(current)
            raise subprocess.CalledProcessError(code, command)
        completed.append(log_name)

    if not args.skip_synth:
        run("synth299.log", "synth_doodles.py", "--per-class", "30000", "--seed", "2026",
            "--ndjson", str(root / "data/synth"), "--workers", str(args.workers))
    if not args.skip_prepare:
        run("prepare299.log", "prepare_synth.py")
    if not args.skip_train:
        run(args.name+".log", "train_head.py", "--corpus", "expanded299-28", "--name", args.name,
            "--epochs", str(args.epochs), "--lr", str(args.lr), "--chunk", str(args.chunk),
            "--baseline", str(out / "wide270_28.pt"), "--extra-hidden", str(args.extra_hidden))
    report = Path(args.report).resolve() if args.report else root / "notes" / (
        "synthetic-comparison-299.json" if args.name == "wide299_28" else f"synthetic-comparison-{args.name}.json")
    run(args.name+".compare.log", "compare_synth.py", "--candidate", args.name, "--out", str(report))
    run(args.name+".export.log", "export.py", "--name", args.name, "--samples", "299",
        "--out-dir", str(root / "luau/models" / args.name))
    # Default export changes only after reviewing retention and false-alarm results.
    status({"phase": "complete", "completed": completed, "candidate": args.name,
            "default_export_unchanged": True})
