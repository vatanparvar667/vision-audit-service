"""Mini benchmark: VLM yes/no scene questions vs. hand labels from the dataset review."""
import sys, time, re, cv2, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL, VDIR = sys.argv[1], sys.argv[2]
QUANT = len(sys.argv) > 3 and sys.argv[3] == "4bit"
KEYS = ["people", "safe_open", "door_open", "cash_on_desk", "drinking", "bag_on_desk", "laptop_app", "phone"]
# None = unknown / not scored
GT = [
 ("X-20", 6.0, dict(people=1, safe_open=1, door_open=0, drinking=0, bag_on_desk=0, phone=0)),
 ("X-20", 0.3, dict(people=1, safe_open=0, door_open=0)),
 ("X-26", 16.5, dict(people=0, safe_open=0, door_open=0, cash_on_desk=1, bag_on_desk=0)),
 ("X-23", 17.5, dict(people=0, safe_open=0, door_open=0, cash_on_desk=0, laptop_app=0)),
 ("X-28", 14.5, dict(people=0, safe_open=0, door_open=0, cash_on_desk=0, laptop_app=1)),
 ("X-32", 8.0, dict(people=1, door_open=1, safe_open=0)),
 ("X-32", 1.0, dict(people=0, door_open=0)),
 ("X-33", 7.0, dict(people=1, door_open=1)),
 ("X-30", 15.0, dict(people=2, door_open=1)),
 ("X-46", 15.5, dict(people=1, drinking=1, cash_on_desk=1, phone=0)),
 ("X-46", 5.0, dict(people=1, drinking=0)),
 ("X-40", 20.0, dict(people=1, bag_on_desk=1, drinking=0)),
 ("X-40", 1.0, dict(people=0, bag_on_desk=0)),
 ("X-38", 33.0, dict(people=2, cash_on_desk=1)),
 ("X-3", 15.0, dict(people=2, safe_open=0, door_open=0, drinking=0, phone=0)),
 ("X-15", 19.5, dict(people=0, bag_on_desk=1)),
 ("X-24", 14.0, dict(people=0, cash_on_desk=0, safe_open=0)),
 ("X-21", 3.0, dict(people=1, safe_open=1)),
 ("X-10", 1.0, dict(people=1, phone=1)),
 ("X-2", 12.5, dict(people=2, phone=0, drinking=0)),
 ("X-31", 1.0, dict(people=0, door_open=0)),
 ("X-16", 18.0, dict(people=2, bag_on_desk=1, door_open=0)),
]
PROMPT = ("CCTV frame of a small bank cash room (cashier desk, metal safe, glass wall with a door). "
          "People seen through the glass wall or reflected in it are OUTSIDE the room - do not count them.\n"
          "Answer 8 items in order, separated by commas, nothing else:\n"
          "1) number of people inside the room (integer)\n"
          "2) is the metal safe's door open? (yes/no)\n"
          "3) is the room's entrance door open? (yes/no)\n"
          "4) are banknotes/cash lying on the desk (not in someone's hands)? (yes/no)\n"
          "5) is someone drinking or eating? (yes/no)\n"
          "6) is a bag, handbag or piece of clothing lying on the desk? (yes/no)\n"
          "7) does the laptop screen show a working application (not dark, not a lock screen)? (yes/no)\n"
          "8) is someone holding a mobile phone? (yes/no)")

def grab(name, t):
    cap = cv2.VideoCapture(f"{VDIR}/{name}.mov"); fps = cap.get(cv2.CAP_PROP_FPS); target = int(round(t * fps)); i = 0
    while True:
        ok, f = cap.read()
        if not ok or i == target: break
        i += 1
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

proc = AutoProcessor.from_pretrained(MODEL)
kw = dict(dtype=torch.bfloat16, device_map="cuda")
if QUANT:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw).eval()
print("vram", round(torch.cuda.memory_allocated() / 1e9, 2), "GB", flush=True)
score = {k: [0, 0] for k in KEYS}; times = []
for name, t, gt in GT:
    img = grab(name, t)
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    times.append(time.time() - t0)
    txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    parts = [p.strip().lower() for p in re.split(r"[,\n]", txt) if p.strip()]
    vals = {}
    for k, p in zip(KEYS, parts):
        p = re.sub(r"^\d\)\s*", "", p)
        if k == "people":
            m = re.search(r"\d+", p); vals[k] = int(m.group()) if m else None
        else:
            vals[k] = 1 if p.startswith("y") else 0 if p.startswith("n") else None
    errs = []
    for k, v in gt.items():
        score[k][1] += 1
        if vals.get(k) == v: score[k][0] += 1
        else: errs.append(f"{k}:{vals.get(k)}!={v}")
    print(f"{name}@{t:<5} {times[-1]:.2f}s  '{txt}'  {'OK' if not errs else 'ERR ' + ' '.join(errs)}", flush=True)
print("accuracy:", {k: f"{a}/{n}" for k, (a, n) in score.items()}, "mean time", round(sum(times) / len(times), 2))
