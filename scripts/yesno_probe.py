import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.vlm import YesNoVLM
vlm = YesNoVLM(Path("models/Qwen3-VL-4B-Instruct"))
Q = "Is the door of the metal safe open?"
for name, ts in [("X-20", [0.3, 3, 6, 9, 11]), ("X-21", [1, 3, 5, 10]), ("X-46", [5, 15]), ("X-10", [5]), ("X-1", [10, 20])]:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    for t in ts:
        i = clip.nearest(t); img = clip.rgb(i)
        t0 = time.time(); p = vlm.yes_prob(img, Q); dt = time.time() - t0
        print(f"{name}@{t}: p_open={p:.2f} ({dt*1000:.0f} ms)", flush=True)
