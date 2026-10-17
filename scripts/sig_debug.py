import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import analyze as A
from app.pipeline.scene import on_desk
labels = json.load(open("runs/labels.json"))
what = sys.argv[1]
names = sys.argv[2:] or sorted((p.stem for p in Path("runs/cache").glob("*.pkl")), key=lambda s: int(s.split("-")[1]))
for name in names:
    if name in ("X-47", "X-48"): continue
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    tl = A.build_timeline(clip, d["coco"], d["owl"])
    tag = "V" if labels[name] else "c"
    if what == "items":
        best = {}
        for i, dets in tl.owl.items():
            for x in dets:
                if x.label in ("bag", "clothes", "cash", "document", "cup", "phone") and on_desk(x.box, tl.layout.desk, 0.6):
                    if x.score > best.get(x.label, (0,))[0]: best[x.label] = (round(x.score, 2), round(float(tl.t[i]), 1))
        cb = {}
        for i, f in enumerate(tl.coco):
            for x in f:
                if x.label in ("handbag", "backpack", "suitcase", "cup", "bottle", "cell phone") and on_desk(x.box, tl.layout.desk, 0.6):
                    if x.score > cb.get(x.label, (0,))[0]: cb[x.label] = (round(x.score, 2), round(float(tl.t[i]), 1))
        print(f"{tag} {name:5} {labels[name]} OWL:{best} COCO:{cb}")
    if what == "door":
        ds = A.door_series(clip, tl)
        print(f"{tag} {name:5} {labels[name]} entries={[(round(float(tl.t[k]),1), tuple(round(v) for v in b) if b else None) for k,b in tl.entries]} exits={[round(float(tl.t[k]),1) for k,_ in tl.exits]}")
        for k, s in ds.items():
            vals = " ".join("." if np.isnan(v) else f"{v:.0f}" for v in s[k:k+60:2])
            print(f"      @{tl.t[k]:.1f}: {vals}")
    if what == "screen":
        if tl.screen:
            occ0 = [i for i in tl.screen if tl.occ[i] == 0]
            occ1 = [i for i in tl.screen if tl.occ[i] > 0]
            f = lambda ids: (round(float(np.median([tl.screen[i][0] for i in ids])),0), round(float(np.median([tl.screen[i][1] for i in ids])),2)) if ids else None
            print(f"{tag} {name:5} {labels[name]} laptop={tuple(round(v) for v in tl.layout.laptop)} present={f(occ1)} empty={f(occ0)}")
        else:
            print(f"{tag} {name:5} no laptop")
