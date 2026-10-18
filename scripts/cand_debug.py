"""Show all rule candidates with pre-verification scores and VLM answers."""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline import analyze as A
labels = json.load(open("runs/labels.json"))
vlm = A.get_vlm()
for name in sys.argv[1:]:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    tl = A.build_timeline(clip, d["coco"], d["owl"])
    sp = A.safe_probs(clip, vlm)
    evs = A.run_rules(clip, tl, sp)
    pre = [(e.type, round(e.t, 1), round(e.score, 2)) for e in evs]
    A.verify(clip, evs, vlm)
    print(f"== {name} gt={labels[name]}")
    for e, p in zip(evs, pre):
        print(f"   {p} -> {round(e.score,2)} vlm={e.facts.get('vlm')} q={[q for _,_,q in e.verify][:1]}")
