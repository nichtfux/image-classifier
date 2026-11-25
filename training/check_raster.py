"""Independent pixel equivalence check for the compiled dual-resolution renderer."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
import raster
import fast_raster
from categories import REAL_CATEGORIES as CATEGORIES
from expand_data import RAW, records

if __name__ == "__main__":
    cases = [s["drawing"] for s in json.loads((RAW.parent / "expanded100/test_strokes.json").read_text())]
    for name in CATEGORIES:
        if (RAW / (name + ".ndjson")).exists():
            cases.extend(d["drawing"] for d in itertools.islice(records(name), 3))
    cases += [[[[0], [0]]], [[[0, 0, 0], [0, 0, 0]]], [[[0, 255], [0, 255]]],
              [[[0, 255], [128, 128]]], [[[128, 128], [0, 255]]]]
    rng = np.random.default_rng(20261109)
    for _ in range(200):
        cases.append([[rng.uniform(-300, 500, n).tolist(), rng.uniform(-300, 500, n).tolist()]
                      for n in rng.integers(1, 30, rng.integers(1, 8))])
    for size in (28, 40):
        ref = raster.Renderer(size)
        for i, drawing in enumerate(cases):
            image = ref.render(drawing)
            if size == 28:
                assert np.array_equal(image, raster.render(drawing)), ("legacy raster changed", i)
            expected = np.round(image * 255).astype(np.uint8)
            actual = fast_raster.render(drawing, size)
            assert np.array_equal(actual, expected), (size, i, np.max(np.abs(actual.astype(int)-expected)))
        t0 = time.perf_counter()
        for drawing in cases:
            fast_raster.render(drawing, size)
        print(f"{size}px: {len(cases)} exact image matches; compiled render {(time.perf_counter()-t0)*1000/len(cases):.3f} ms/image")
