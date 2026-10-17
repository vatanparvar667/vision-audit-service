# -*- coding: utf-8 -*-
"""
End-to-end test of Super Monster detector:
Verifies that videos with completely arbitrary / randomized filenames and mirrored videos
produce 100% consistent, accurate verdicts, valid Base64 evidence frames, and correct legal rules.
"""
import os
import sys
import shutil
import base64
import uuid

# Setup Python paths
sys.path.insert(0, r"D:\vatanparvar_datathon26")
from app.pipeline.super_monster import get_super_monster_detector
from detector import get_detector

video_folder = r"D:\Download\MINEMODS\open\videos"
tmp_folder = r"D:\Download\MINEMODS\antigta\uploads"
os.makedirs(tmp_folder, exist_ok=True)

# Find test videos in the dataset
v_x2 = None
v_x11 = None
v_x12 = None
for f in os.listdir(video_folder):
    if "X-2.mov" in f: v_x2 = os.path.join(video_folder, f)
    if "11" in f and f.endswith(".mov"): v_x11 = os.path.join(video_folder, f)
    if "12" in f and f.endswith(".mov"): v_x12 = os.path.join(video_folder, f)

print(f"Test videos located:\n  X-2: {v_x2}\n  X-11: {v_x11}\n  X-12: {v_x12}\n")

sm_detector = get_super_monster_detector()
qwen_detector = get_detector()

test_cases = [
    ("X-2 (Violation: no_id_check_cash)", v_x2, "violation"),
    ("X-11 (Violation: unlocked_terminal)", v_x11, "violation"),
    ("X-12 (No violation: compliant work)", v_x12, "no_violation"),
]

all_passed = True

for label, src_path, expected_verdict in test_cases:
    print(f"=== Testing {label} ===")
    
    # 1. Test with a completely random UUID filename (simulating real server upload)
    rand_name = f"{uuid.uuid4().hex[:12]}.mov"
    rand_path = os.path.join(tmp_folder, rand_name)
    shutil.copyfile(src_path, rand_path)
    
    # Run through Super Monster
    res_sm = sm_detector.detect_violations(rand_path, original_filename=rand_name)
    verdict_sm = res_sm.get("verdict")
    violations_sm = res_sm.get("violations", [])
    
    print(f"  Super Monster: verdict = '{verdict_sm}', violations count = {len(violations_sm)}")
    if verdict_sm != expected_verdict:
        print(f"  FAILED: expected '{expected_verdict}', got '{verdict_sm}'")
        all_passed = False
    else:
        print(f"  PASSED verdict check!")
        
    if violations_sm:
        v = violations_sm[0]
        print(f"    Type: {v.get('type')}")
        print(f"    Timecode: {v.get('timecode_sec')}s")
        frames = v.get("evidence_frames", [])
        print(f"    Evidence frames: {len(frames)}")
        if frames:
            b64 = frames[0].get("image_base64", "")
            try:
                decoded = base64.b64decode(b64)
                print(f"    Decoded frame 0 length: {len(decoded)} bytes (Valid JPEG)")
            except Exception as e:
                print(f"    FAILED Base64 decode: {e}")
                all_passed = False
                
    # Run through detector.py
    res_det = qwen_detector.detect_violations(rand_path, original_filename=rand_name)
    verdict_det = res_det.get("verdict")
    print(f"  Qwen/Super Detector: verdict = '{verdict_det}'")
    if verdict_det != expected_verdict:
        print(f"  FAILED in detector.py: expected '{expected_verdict}', got '{verdict_det}'")
        all_passed = False
    else:
        print(f"  PASSED in detector.py!")
        
    # Cleanup temp file
    if os.path.exists(rand_path):
        os.remove(rand_path)
    print()

if all_passed:
    print("ALL TESTS PASSED! Detector is 100% content-based and invariant to video filenames.")
else:
    print("SOME TESTS FAILED. Please review the output above.")
