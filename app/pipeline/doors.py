"""Door-left-open detection without knowing where the door is.

Idea: when somebody enters or leaves, the door panel moves. If the door is closed again, the pixels around the
entry path return to how they looked before; if it stays open, a large region next to the path stays changed.
People are masked out, so only static scene changes count. Works for any camera angle with a static camera.
"""

import cv2
import numpy as np

from app.pipeline.perception import Det
from app.pipeline.video import VideoClip

DIFF_THR = 22          # grey-level difference that counts as "changed"
PERSIST = 0.75         # share of observations a pixel must stay changed
MIN_BLOB = 0.010       # changed blob area as a share of the frame -> door open


def _person_mask(boxes, shape, sx, dil=4):
    m = np.zeros(shape, dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v * sx)) for v in b]
        m[max(0, y0 - dil):max(0, y1 + dil), max(0, x0 - dil):max(0, x1 + dil)] = True
    return m


def door_change(clip: VideoClip, coco: list[list[Det]], k: int, fps: float,
                path_boxes: list[tuple], desk=None, ref_idx: list[int] | None = None,
                post_end: int | None = None) -> dict | None:
    """Persistent change near the entry path after event at sample k vs. reference frames ref_idx
    (the scene before anybody came to the door). Returns blob info or None."""
    n = len(clip.samples)
    small_h, small_w = clip.samples[0].small.shape
    sx = small_w / clip.width
    raw = [[d.box for d in f if d.label == "person" and d.score >= 0.3] for f in coco]
    if not ref_idx or len(ref_idx) < 2:
        return None
    # reference: per-pixel median of unoccluded observations
    stack = np.stack([clip.samples[i].small for i in ref_idx]).astype(np.float32)
    masks = np.stack([_person_mask(raw[i], (small_h, small_w), sx) for i in ref_idx])
    stack_masked = np.where(masks, np.nan, stack)
    with np.errstate(all="ignore"):
        ref = np.nanmedian(stack_masked, axis=0)
    ref = np.where(np.isnan(ref), np.median(stack, axis=0), ref)

    end = min(n, k + int(12 * fps)) if post_end is None else min(n, post_end)
    post = [i for i in range(k + int(2.5 * fps), end)]
    if len(post) < int(3 * fps):
        return None
    changed = np.zeros((small_h, small_w), dtype=np.float32)
    seen = np.zeros((small_h, small_w), dtype=np.float32)
    for i in post:
        m = ~_person_mask(raw[i], (small_h, small_w), sx)
        d = np.abs(clip.samples[i].small.astype(np.float32) - ref) > DIFF_THR
        changed += (d & m)
        seen += m
    ok = seen >= 0.5 * len(post)
    persist = np.zeros_like(ok)
    persist[ok] = changed[ok] / seen[ok] >= PERSIST

    # region of interest: around the entry path, from the top of the frame down to the path bottom
    if not path_boxes:
        return None
    px0 = min(b[0] for b in path_boxes); px1 = max(b[2] for b in path_boxes)
    py1 = max(b[3] for b in path_boxes)
    w = px1 - px0
    rx0, rx1 = max(0, px0 - 1.0 * w), min(clip.width, px1 + 1.0 * w)
    ry0, ry1 = 0, py1
    if desk is not None:
        ry1 = min(ry1, desk[1] + 0.15 * (desk[3] - desk[1]))
    roi = np.zeros_like(persist)
    roi[int(ry0 * sx):int(ry1 * sx), int(rx0 * sx):int(rx1 * sx)] = True
    cand = (persist & roi).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if num <= 1:
        return {"area": 0.0, "box": None}
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[j]
    box = (x / sx, y / sx, (x + bw) / sx, (y + bh) / sx)
    return {"area": float(area) / (small_w * small_h), "box": box}
