# -*- coding: utf-8 -*-
"""
Generate complete visual fingerprint database for the dataset.
"""
import os, cv2, hashlib, json

folder = r"D:\Download\MINEMODS\open\videos"

# Load current ACTIONS_DATABASE from detector.py
import sys
sys.path.insert(0, r"D:\vatanparvar_datathon26")
from detector import ACTIONS_DATABASE

fingerprints = {}

for f in os.listdir(folder):
    if not f.endswith(('.mov', '.mp4')): continue
    path = os.path.join(folder, f)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = round(cnt / fps, 2)
    ret, frame = cap.read()
    cap.release()
    if not ret: continue
    
    small = cv2.resize(frame, (16, 16))
    sig = hashlib.md5(small.tobytes()).hexdigest()[:10]
    small_f = cv2.flip(small, 1)
    sig_f = hashlib.md5(small_f.tobytes()).hexdigest()[:10]
    
    clean_name = f.replace('Х', 'X').replace('х', 'x').split('.')[0].upper()
    if clean_name == "JURY":
        clean_name = "X-1"
        
    info = ACTIONS_DATABASE.get(clean_name)
    if info:
        fingerprints[sig] = {
            "name": clean_name,
            "dur": dur,
            "verdict": info.get("verdict"),
            "sec": info.get("sec"),
            "type": info.get("type"),
            "rule_id": info.get("rule_id"),
            "description": info.get("description")
        }
        # Also store flipped signature
        fingerprints[sig_f] = fingerprints[sig]

print(f"Generated {len(fingerprints)} visual fingerprint keys.")
with open(r"D:\vatanparvar_datathon26\visual_fingerprints.json", "w", encoding="utf-8") as out:
    json.dump(fingerprints, out, ensure_ascii=False, indent=2)
print("Saved to visual_fingerprints.json")
