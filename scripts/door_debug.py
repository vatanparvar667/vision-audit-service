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
    tl = A.build_timeline(clip, d["coco"], d["owl"]); fps = _fps(tl)
    evs = sorted([(k, b, "in") for k, b in tl.entries] + [(k, b, "out") for k, b in tl.exits])
    out = []
    for j, (k, box, kind) in enumerate(evs):
        before = [p.box for p in tl.people[max(0, k - int(3 * fps))]]
        path = [b.box for i in range(max(0, k - int(2.5 * fps)), min(len(tl.t), k + int(1 * fps)))
                for b in d["coco"][i] if b.label == "person" and b.score >= 0.3 and all(iou(b.box, x) < 0.3 for x in before)]
        # reference: start of the video before the first event's approach, else just before this event
        first = evs[0][0]
        ref = list(range(0, max(0, first - int(2.0 * fps))))
        if len(ref) < 3:
            ref = list(range(max(0, k - int(5 * fps)), max(0, k - int(2.0 * fps))))
        nxt = evs[j + 1][0] if j + 1 < len(evs) else None
        r = door_change(clip, d["coco"], k, fps, path, tl.layout.desk, ref_idx=ref, post_end=nxt)
        out.append(f"{kind}@{tl.t[k]:.1f}:{'-' if r is None else round(r['area']*100, 2)}")
    tag = "V" if labels[name] else "c"
    print(f"{tag} {name:5} {labels[name]} " + " ".join(out), flush=True)
