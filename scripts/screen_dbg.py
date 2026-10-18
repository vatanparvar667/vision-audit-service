import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import analyze as A
from app.pipeline.rules import runs
labels = json.load(open("runs/labels.json"))
for name in ["X-11","X-12","X-13","X-15","X-22","X-23","X-24","X-25","X-26","X-27","X-28","X-29","X-40"]:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    tl = A.build_timeline(clip, d["coco"], d["owl"])
    for a, b in runs(tl.occ == 0):
        if a == 0: continue
        ids = [i for i in range(a, b + 1) if i in tl.screen]
        if not ids: continue
        sv = np.median([tl.screen[i][0] for i in ids]); wh = np.median([tl.screen[i][1] for i in ids]); gm = np.median([tl.brightness[i] for i in ids])
        print(f"{'V' if labels[name] else 'c'} {name} empty {tl.t[a]:.1f}-{tl.t[b]:.1f} screen_v={sv:.0f} white={wh:.2f} frame={gm:.0f} ratio={sv/gm:.2f}")
