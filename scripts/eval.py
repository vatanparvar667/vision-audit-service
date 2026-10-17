"""Offline evaluation on cached perception: verdict accuracy and timecode error vs. labels.
Usage: python -I scripts/eval.py [--vlm] [names...]"""
import json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app.pipeline import analyze as A
from app.pipeline.report import MIN_SCORE

args = [a for a in sys.argv[1:] if not a.startswith("--")]
use_vlm = "--vlm" in sys.argv
labels = json.load(open("runs/labels.json"))
names = args or sorted((p.stem for p in Path("runs/cache").glob("*.pkl")), key=lambda s: int(s.split("-")[1]))
names = [n for n in names if n not in ("X-47", "X-48")]  # byte-identical duplicates of X-4 / X-5
tp = fp = tn = fn = 0; errs = []
for name in names:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    t0 = time.time()
    if use_vlm:
        res = A.analyze_clip(clip, perception={"coco": d["coco"], "owl": d["owl"]})
        evs = [(v["type_id"], v["timecode_sec"], v["score"]) for v in res["violations"]]
    else:
        tl = A.build_timeline(clip, d["coco"], d["owl"])
        evs = [(e.type, round(e.t, 1), round(e.score, 2)) for e in A.run_rules(clip, tl, {}) if e.score >= MIN_SCORE]
    gt = labels[name]
    pred = bool(evs)
    if gt and pred: tp += 1
    elif gt and not pred: fn += 1
    elif not gt and pred: fp += 1
    else: tn += 1
    terr = []
    for g in gt:
        if evs:
            terr.append(min(abs(e[1] - g) for e in evs))
    errs += terr
    mark = "OK " if bool(gt) == pred else ("MISS" if gt else "FP  ")
    print(f"{mark} {name:5} gt={gt} pred={[(e[0], e[1], e[2]) for e in evs]} terr={[round(x,1) for x in terr]} {time.time()-t0:.1f}s", flush=True)
print(f"\nverdict acc {(tp+tn)}/{len(names)}  TP={tp} FN={fn} FP={fp} TN={tn}  median terr={np.median(errs) if errs else None}  within3s={sum(e<=3 for e in errs)}/{len(errs)}")
