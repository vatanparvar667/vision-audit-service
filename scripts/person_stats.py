import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for name in sys.argv[1:]:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]; H = clip.height
    print("==", name, clip.width, H)
    for i in range(0, len(clip.samples), 4):
        ps = [x for x in d["coco"][i] if x.label == "person" and x.score >= 0.4]
        print(f"{clip.samples[i].t:5.1f} " + " ".join(f"[{p.score:.2f} h={p.h/H:.2f} x{p.cx:.0f} y{p.box[3]:.0f}]" for p in sorted(ps, key=lambda p: -p.h)))
