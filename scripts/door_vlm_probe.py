import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.vlm import YesNoVLM
vlm = YesNoVLM(Path("models/Qwen3-VL-4B-Instruct"))
QS = ["Is the glass door of this room standing open?",
      "Look at the glass wall with the door. Is the door closed?"]
cases = [("X-32", 1.0, 0), ("X-32", 8.0, 1), ("X-32", 15.0, 1), ("X-33", 1.0, 0), ("X-33", 8.0, 1), ("X-30", 1.0, 0), ("X-30", 8.0, 1), ("X-30", 15.0, 1),
         ("X-37", 1.0, 0), ("X-37", 10.0, 0), ("X-35", 2.0, 0), ("X-35", 14.0, 0), ("X-31", 15.0, 0), ("X-19", 20.0, 0), ("X-22", 20.0, 0), ("X-3", 10.0, 0)]
for name, t, gt in cases:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    img = clip.rgb(clip.nearest(t)); crop = img[0:470, 0:300]
    ps = [vlm.yes_prob(crop, QS[0]), 1 - vlm.yes_prob(crop, QS[1]), vlm.yes_prob(img, QS[0])]
    print(f"{name}@{t} gt_open={gt}  crop_open={ps[0]:.2f} crop_notclosed={ps[1]:.2f} full_open={ps[2]:.2f}", flush=True)
