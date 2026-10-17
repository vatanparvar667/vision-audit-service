"""Probe RT-DETRv2 COCO detector on a video: per-second summary of persons and key objects."""
import sys, time, cv2, torch, numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL, VIDEO = sys.argv[1], sys.argv[2]
FPS_S = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForObjectDetection.from_pretrained(MODEL).to("cuda").eval().half()
id2label = model.config.id2label
KEEP = {"person", "cup", "bottle", "cell phone", "handbag", "backpack", "suitcase", "laptop", "book", "chair", "dining table", "tv", "wine glass", "bowl", "sandwich", "donut", "cake", "banana", "remote", "keyboard", "mouse", "scissors", "umbrella", "tie", "clock", "refrigerator", "microwave", "oven", "toaster", "sink"}
cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS)
step = int(round(fps / FPS_S)); frames = []; times = []; i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % step == 0:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); times.append(i / fps)
    i += 1
H, W = frames[0].shape[:2]
t0 = time.time(); res = []
for b in range(0, len(frames), 8):
    batch = frames[b:b + 8]
    inp = proc(images=batch, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(pixel_values=inp["pixel_values"].half())
    out.logits = out.logits.float(); out.pred_boxes = out.pred_boxes.float()
    res += proc.post_process_object_detection(out, threshold=0.3, target_sizes=[(H, W)] * len(batch))
dt = time.time() - t0
print(f"{len(frames)} frames in {dt:.2f}s ({dt/len(frames)*1000:.0f} ms/frame)  size {W}x{H}")
for t, r in zip(times, res):
    items = []
    for s, l, bx in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
        name = id2label[l]
        if name not in KEEP: continue
        x0, y0, x1, y1 = bx
        items.append(f"{name}:{s:.2f}[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")
    print(f"{t:5.1f}s " + " ".join(items))
