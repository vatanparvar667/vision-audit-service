"""Recompute the OWLv2 detections of an existing perception cache with the current detector settings
(COCO detections and decoded frames are reused). Usage: python -I scripts/rebuild_owl.py SRC_DIR DST_DIR"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import analyze as A  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
per = A.get_perception()
for p in sorted(src.glob("*.pkl")):
    out = dst / p.name
    if out.exists():
        continue
    d = pickle.load(p.open("rb"))
    t0 = time.time()
    d["owl"] = per.detect_owl(d["clip"], sorted(d["owl"]))
    pickle.dump(d, out.open("wb"))
    print(f"{p.stem}: {time.time() - t0:.1f}s", flush=True)
