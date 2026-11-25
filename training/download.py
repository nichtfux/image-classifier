"""Download the first part of each category's *simplified* stroke file from Google's
official Quick, Draw! bucket (CC BY 4.0). Only the first BYTES of each file are
fetched with an HTTP Range request, which is plenty for training a small net.

    python download.py            # -> ../data/simplified/<category>.ndjson
"""
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from categories import REAL_CATEGORIES as CATEGORIES

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
BYTES = 16 * 1024 * 1024  # first 16 MB per category (~25-35k drawings)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "simplified")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in CATEGORIES:
        path = os.path.join(OUT, name + ".ndjson")
        if os.path.exists(path):
            print("have", name)
            continue
        url = BASE + urllib.parse.quote(name) + ".ndjson"
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{BYTES - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        # drop the last (cut) line
        data = data[: data.rfind(b"\n") + 1]
        with open(path, "wb") as f:
            f.write(data)
        print("got", name, len(data) // 1024, "KB", data.count(b"\n"), "drawings")


if __name__ == "__main__":
    main()
