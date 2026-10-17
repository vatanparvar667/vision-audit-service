import pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.perception import nms_by_label
def count(frame, W, H, smin):
    ps = nms_by_label([d for d in frame if d.label == "person" and d.score >= smin], 0.45)
    n = 0
    for p in ps:
        a = p.w * p.h / (W * H); bot = p.box[3] / H
        if a >= 0.022 or (a >= 0.014 and bot >= 0.42): n += 1
    return n
for smin in (0.6, 0.75):
    print("##### smin", smin)
    for name in sys.argv[1:]:
        d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]; W, H = clip.width, clip.height
        raw = np.array([count(f, W, H, smin) for f in d["coco"]])
        k = 7; p = np.pad(raw, k // 2, mode="edge")
        sm = np.array([int(np.median(p[i:i + k])) for i in range(len(raw))])
        ts = [s.t for s in clip.samples]; segs = []; cur = None
        for i, o in enumerate(sm):
            if cur is None or o != cur[0]: cur = [o, ts[i], ts[i]]; segs.append(cur)
            else: cur[2] = ts[i]
        print(f"{name}: " + " ".join(f"{o}@{a:.1f}-{b:.1f}" for o, a, b in segs))
