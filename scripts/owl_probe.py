"""Probe OWLv2 open-vocabulary detection on selected frames."""
import sys, time, cv2, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL, VDIR = sys.argv[1], sys.argv[2]
cases = [("X-20", 6.0), ("X-26", 16.5), ("X-32", 6.0), ("X-40", 20.0), ("X-38", 33.0), ("X-3", 15.0), ("X-24", 14.0), ("X-46", 15.5)]
queries = ["a safe", "an open safe door", "a desk", "a banknote counting machine", "banknotes", "a stack of paper money", "a jacket", "a handbag", "an open door", "a glass door", "a laptop", "a coffee cup", "a mobile phone", "a document folder"]
proc = Owlv2Processor.from_pretrained(MODEL)
model = Owlv2ForObjectDetection.from_pretrained(MODEL).to("cuda").eval()

def grab(name, t):
    cap = cv2.VideoCapture(f"{VDIR}/{name}.mov"); fps = cap.get(cv2.CAP_PROP_FPS); target = int(round(t * fps)); i = 0
    while True:
        ok, f = cap.read()
        if not ok or i == target: break
        i += 1
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

for name, t in cases:
    img = grab(name, t)
    inp = proc(text=[queries], images=img, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model(**inp)
    W, H = img.size; S = max(W, H)
    res = proc.post_process_grounded_object_detection(out, threshold=0.12, target_sizes=torch.tensor([[S, S]]).cuda())[0]
    dt = time.time() - t0
    best = {}
    for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
        q = queries[l]
        best.setdefault(q, []).append((round(s, 2), [round(v) for v in b]))
    print(f"== {name}@{t} {dt*1000:.0f}ms")
    for q in queries:
        if q in best:
            print("   ", q, sorted(best[q], reverse=True)[:3])
