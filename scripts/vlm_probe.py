"""Probe Qwen3-VL on single frames: speed + answers to scene-state questions."""
import sys, time, json, cv2, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = sys.argv[1]
VID = Path(sys.argv[2])
cases = [("X-20", 6.0), ("X-26", 16.5), ("X-32", 6.0), ("X-46", 15.5), ("X-40", 20.0), ("X-23", 17.5), ("X-28", 12.0), ("X-3", 15.0), ("X-30", 14.0), ("X-2", 12.5)]

def grab(name, t):
    cap = cv2.VideoCapture(str(VID / f"{name}.mov")); fps = cap.get(cv2.CAP_PROP_FPS)
    target = int(round(t * fps)); i = 0
    while True:
        ok, f = cap.read()
        if not ok or i == target: break
        i += 1
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

proc = AutoProcessor.from_pretrained(MODEL)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
print("load", round(time.time() - t0, 1), "s; vram", round(torch.cuda.memory_allocated() / 1e9, 2), "GB", flush=True)

Q = ("This is a CCTV frame of a bank cash room. Answer with JSON only, keys: "
     '"people_in_room" (int, people physically inside this room, ignore people behind the glass wall), '
     '"cashier_seated_at_desk" (bool), "safe_door_open" (bool), "room_door_open" (bool), '
     '"banknotes_visible_on_desk" (bool), "person_drinking_or_eating" (bool), '
     '"handbag_or_clothes_on_desk" (bool), "laptop_screen_shows_work_app" (bool), "phone_in_hand" (bool), '
     '"description" (one short sentence).')
for name, t in cases:
    img = grab(name, t)
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": Q}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"== {name}@{t}s  tokens_in={inputs['input_ids'].shape[1]}  {time.time()-t0:.2f}s\n{txt}", flush=True)
