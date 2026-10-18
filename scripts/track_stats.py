import pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.scene import track_people
for name in sorted(p.stem for p in Path("runs/cache").glob("*.pkl")):
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]; ts = [s.t for s in clip.samples]
    tr = [t for t in track_people(d["coco"], clip.height) if len(t.boxes) >= 3]
    out = []
    for t in tr:
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in t.boxes.values()]
        bottoms = [b[3] for b in t.boxes.values()]
        out.append(f"T{t.tid}[{ts[t.first]:.0f}-{ts[t.last]:.0f}s pk{t.peak():.2f} md{t.median_score():.2f} A{np.median(areas)/1000:.1f}k bot{np.median(bottoms):.0f} x{np.median([(b[0]+b[2])/2 for b in t.boxes.values()]):.0f}]")
    print(name, " ".join(out))
