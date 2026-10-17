import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import analyze as A
from app.pipeline.doors import door_change
from app.pipeline.perception import iou
from app.pipeline.rules import _fps
labels = json.load(open("runs/labels.json"))
names = sys.argv[1:] or sorted((p.stem for p in Path("runs/cache").glob("*.pkl")), key=lambda s: int(s.split("-")[1]))
for name in names:
    if name in ("X-47", "X-48"): continue
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    tl = A.build_timeline(clip, d["coco"], d["owl"]); fps = _fps(tl); n = len(tl.t)
    evs = sorted([k for k, b in tl.entries] + [k for k, b in tl.exits])
    if not evs:
        print(f"{'V' if labels[name] else 'c'} {name:5} {labels[name]} no events"); continue
    path = []
    for k in evs:
        before = [p.box for p in tl.people[max(0, k - int(3 * fps))]]
        path += [b.box for i in range(max(0, k - int(2.5 * fps)), min(n, k + int(1 * fps)))
                 for b in d["coco"][i] if b.label == "person" and b.score >= 0.3 and all(iou(b.box, x) < 0.3 for x in before)]
    ref = list(range(0, min(3, n)))
    k_last = n - int(3.5 * fps)
    r = door_change(clip, d["coco"], k_last - int(2.5 * fps), fps, path, tl.layout.desk, ref_idx=ref, post_end=n)
    print(f"{'V' if labels[name] else 'c'} {name:5} {labels[name]} end-vs-start blob={'-' if r is None else round(r['area']*100, 2)} box={None if not r or not r['box'] else tuple(round(v) for v in r['box'])}", flush=True)
