"""Add the desk-crop OWLv2 pass to an existing perception cache (COCO detections are reused).
Usage: python -I scripts/upgrade_cache.py SRC_DIR DST_DIR"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import analyze as A  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob("*.pkl")):
    out = dst / p.name
    if out.exists():
        continue
    d = pickle.load(p.open("rb"))
    t0 = time.time()
    d["owl"] = A.add_desk_pass(d["clip"], d["coco"], d["owl"], sorted(d["owl"]))
    pickle.dump(d, out.open("wb"))
    print(f"{p.stem}: {time.time() - t0:.1f}s", flush=True)
