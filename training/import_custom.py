"""Turn drawings from the Doodle Collector page into Quick, Draw!-style ndjson.

The collector (a small web page, not included in this repository)
stores one document per drawing at drawings/<person id>/items/<doc id>:
    {cls, v, at, ms, pad, drawing: [[xs], [ys]] per stroke (simplified, 0-255), raw: [[xs], [ys], [ms]] per stroke}

Claude downloads them with the ArtifactData tool (list with out_dir) into data/custom/raw/,
which gives one JSON file per drawing: {"id", "data": {...}, "version", "updatedAt"} or the bare body.
This script collects those files and writes data/custom/<class>.ndjson in the same format as Google's
simplified files ({"word", "key_id", "recognized", "drawing"}), so expand_data.py can treat custom
classes like any other.

    python -I import_custom.py [--raw ../data/custom/raw] [--out ../data/custom]
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    body = doc.get("data", doc) if isinstance(doc, dict) else None
    if not isinstance(body, dict) or body.get("probe"):
        return None, None
    return (doc.get("id") or Path(path).stem), body


def valid(drawing):
    if not isinstance(drawing, list) or not drawing:
        return False
    for stroke in drawing:
        if (not isinstance(stroke, list) or len(stroke) < 2 or len(stroke[0]) != len(stroke[1])
                or not stroke[0] or not all(isinstance(v, int) and 0 <= v <= 255 for v in stroke[0] + stroke[1])):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE.parent / "data" / "custom" / "raw"))
    ap.add_argument("--out", default=str(HERE.parent / "data" / "custom"))
    a = ap.parse_args()
    by_class = defaultdict(list)
    skipped = 0
    for root, _, files in os.walk(a.raw):
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            key, body = load(os.path.join(root, name))
            if body is None or not valid(body.get("drawing")) or not isinstance(body.get("cls"), str):
                skipped += 1
                continue
            by_class[body["cls"]].append({"word": body["cls"], "key_id": key, "recognized": True,
                                          "drawing": [[s[0], s[1]] for s in body["drawing"]],
                                          "ms": body.get("ms")})
    os.makedirs(a.out, exist_ok=True)
    for cls, rows in sorted(by_class.items()):
        with open(os.path.join(a.out, f"{cls}.ndjson"), "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"{cls:16s} {len(rows):6d}")
    print(f"classes {len(by_class)}, drawings {sum(map(len, by_class.values()))}, skipped {skipped}")


if __name__ == "__main__":
    main()
