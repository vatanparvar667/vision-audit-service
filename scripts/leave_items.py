"""Detections on the desk while the room is empty after the cashier left (leaving videos only)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import analyze as A
from app.pipeline.rules import runs
from app.pipeline.scene import on_desk
labels = json.load(open("runs/labels.json"))
SKIP = {"person", "chair", "dining table", "laptop", "tv", "keyboard"}
for name in ["X-11","X-12","X-13","X-15","X-22","X-23","X-24","X-25","X-26","X-27","X-28","X-29"]:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    tl = A.build_timeline(clip, d["coco"], d["owl"])
    for a, b in runs(tl.occ == 0):
        if a == 0: continue
        coco = {}; owl = {}
        for i in range(a, b + 1):
            for x in tl.coco[i]:
                if x.label not in SKIP and on_desk(x.box, tl.layout.desk, 0.6) and x.score >= 0.3:
                    k = (x.label, tuple(int(v // 20) for v in x.box[:2]))
                    coco[k] = max(coco.get(k, 0), round(x.score, 2))
            if i in tl.owl:
                for x in tl.owl[i]:
                    if x.label not in ("desk", "door", "safe", "machine") and on_desk(x.box, tl.layout.desk, 0.6) and x.score >= 0.15:
                        k = (x.label, tuple(int(v // 20) for v in x.box[:2]))
                        owl[k] = max(owl.get(k, 0), round(x.score, 2))
        print(f"{'V' if labels[name] else 'c'} {name} empty {tl.t[a]:.1f}-{tl.t[b]:.1f} COCO={sorted(coco.items(), key=lambda kv: -kv[1])[:6]} OWL={sorted(owl.items(), key=lambda kv: -kv[1])[:6]}")
