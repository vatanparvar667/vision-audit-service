# -*- coding: utf-8 -*-
"""
Тестирование детектора на выборке видеороликов из датасета ForteBank.
"""
import os
import sys
import pandas as pd
from detector import get_detector

sys.stdout.reconfigure(encoding='utf-8')

labels_file = r"D:\Download\MINEMODS\open\open_labels.xlsx"
videos_dir = r"D:\Download\MINEMODS\open\videos"

df = pd.read_excel(labels_file)
detector = get_detector()

total = 0
correct_verdicts = 0
results_summary = []

print(f"{'Видео':<10} | {'Истина':<15} | {'Предикт':<15} | {'Время':<8} | {'Статус':<8}")
print("-" * 65)

# Протестируем все 42 видеоролика
for idx, row in df.iterrows():
    raw_name = str(row.iloc[0]).strip()
    gt_label = str(row.iloc[2]).strip()
    gt_ts = str(row.iloc[1]).strip()
    
    gt_verdict = "violation" if ("нарушение" in gt_label and "без" not in gt_label) else "no_violation"

    candidates = [
        f for f in os.listdir(videos_dir)
        if f.lower().startswith(raw_name.lower().replace('х', 'x')) or f.lower().startswith(raw_name.lower())
    ]
    if not candidates:
        continue

    vpath = os.path.join(videos_dir, candidates[0])
    res = detector.detect_violations(vpath)
    pred_verdict = res["verdict"]

    is_correct = (pred_verdict == gt_verdict)
    total += 1
    if is_correct:
        correct_verdicts += 1

    status_str = "OK" if is_correct else "MISMATCH"
    time_str = f"{res.get('processing_time_sec', 0.0):.2f}s"

    print(f"{raw_name:<10} | {gt_verdict:<15} | {pred_verdict:<15} | {time_str:<8} | {status_str:<8}")
    
    for v in res.get("violations", []):
        print(f"   -> [Обнаружено] {v['type']} на {v['timecode_sec']}с (доказательств: {len(v['evidence_frames'])} кадров)")

accuracy = (correct_verdicts / total) * 100 if total > 0 else 0
print("-" * 65)
print(f"Точность определения вердикта (Критерий 1): {correct_verdicts}/{total} ({accuracy:.1f}%)")
