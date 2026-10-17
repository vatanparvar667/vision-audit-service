"""Scene understanding over time: person tracks, room occupancy, layout zones, persistent visual changes."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.pipeline.perception import Det, iou, overlap_frac
from app.pipeline.video import VideoClip

# A person counts as "in the room" when the detector is confident: people seen through the glass wall,
# their reflections and passers-by are blurred and get noticeably lower scores (checked on the dataset).
PERSON_MIN_SCORE = 0.5
TRACK_MIN_PEAK = 0.85


@dataclass
class Track:
    tid: int
    boxes: dict[int, tuple] = field(default_factory=dict)    # sample idx -> box
    scores: dict[int, float] = field(default_factory=dict)

    @property
    def first(self) -> int:
        return min(self.boxes)

    @property
    def last(self) -> int:
        return max(self.boxes)

    def peak(self) -> float:
        return max(self.scores.values())

    def median_score(self) -> float:
        return float(np.median(list(self.scores.values())))

    def box_at(self, i: int):
        if i in self.boxes:
            return self.boxes[i]
        keys = sorted(self.boxes)
        j = min(keys, key=lambda k: abs(k - i))
        return self.boxes[j]


def track_people(dets: list[list[Det]], frame_h: float, max_gap: int = 8) -> list[Track]:
    """Greedy IoU/centre-distance tracker. max_gap in samples (~2 s at 4 fps)."""
    tracks: list[Track] = []
    active: list[Track] = []
    nid = 0
    for i, frame in enumerate(dets):
        people = [d for d in frame if d.label == "person" and d.score >= PERSON_MIN_SCORE]
        people.sort(key=lambda d: -d.score)
        used = set()
        # match by IoU first, then by centre distance
        pairs = []
        for ti, tr in enumerate(active):
            last_box = tr.boxes[tr.last]
            for pi, p in enumerate(people):
                o = iou(last_box, p.box)
                cdist = np.hypot((last_box[0] + last_box[2]) / 2 - p.cx, (last_box[1] + last_box[3]) / 2 - p.cy)
                gap = i - tr.last
                if o > 0.2 or cdist < (0.08 + 0.02 * gap) * frame_h:
                    pairs.append((-(o + 0.5 * (1 - cdist / frame_h)), ti, pi))
        pairs.sort()
        matched_t = set()
        for _, ti, pi in pairs:
            if ti in matched_t or pi in used:
                continue
            tr = active[ti]
            tr.boxes[i] = people[pi].box
            tr.scores[i] = people[pi].score
            matched_t.add(ti)
            used.add(pi)
        for pi, p in enumerate(people):
            if pi in used:
                continue
            # skip near-duplicate boxes of an already matched person
            if any(iou(p.box, active[t].boxes.get(i, (0, 0, 0, 0))) > 0.5 for t in matched_t):
                continue
            tr = Track(nid, {i: p.box}, {i: p.score})
            nid += 1
            tracks.append(tr)
            active.append(tr)
        active = [t for t in active if i - t.last <= max_gap]
    return tracks


def in_room_tracks(tracks: list[Track], width: int, height: int, min_len: int = 3) -> list[Track]:
    """People physically in the room. Passers-by behind the glass wall and reflections in it are confidently
    detected too, but they are smaller and stand higher in the frame (further away)."""
    frame = float(width * height)
    out = []
    for t in tracks:
        if len(t.boxes) < min_len or t.peak() < 0.8:
            continue
        areas = [(b[2] - b[0]) * (b[3] - b[1]) / frame for b in t.boxes.values()]
        area = float(np.median(areas))
        bottom = float(np.median([b[3] for b in t.boxes.values()])) / height
        if area >= 0.022 or (area >= 0.014 and bottom >= 0.42):
            out.append(t)
    return out


ROOM_PERSON_SCORE = 0.75


def room_people(frame: list[Det], width: int, height: int) -> list[Det]:
    """Confident person detections that are physically in the room (see in_room_tracks for the criterion)."""
    from app.pipeline.perception import nms_by_label

    out = []
    for p in nms_by_label([d for d in frame if d.label == "person" and d.score >= ROOM_PERSON_SCORE], 0.45):
        a = p.w * p.h / (width * height)
        if a >= 0.022 or (a >= 0.014 and p.box[3] / height >= 0.42):
            out.append(p)
    return out


def rolling_median(x: np.ndarray, k: int = 7) -> np.ndarray:
    p = np.pad(x, k // 2, mode="edge")
    return np.array([int(np.median(p[i:i + k])) for i in range(len(x))])


def occupancy_changes(people: list[list[Det]], occ: np.ndarray, look: int = 3):
    """Entries/exits from changes of the smoothed head count. Returns lists of (sample, box of the person)."""
    entries, exits = [], []
    n = len(occ)
    for i in range(1, n):
        if occ[i] > occ[i - 1]:
            before = [d.box for j in range(max(0, i - look - 2), max(0, i - 1)) for d in people[j]]
            cands = [d for j in range(i - 1, min(n, i + look)) for d in people[j]
                     if all(iou(d.box, b) < 0.3 for b in before)]
            box = max(cands, key=lambda d: d.score).box if cands else None
            entries.append((i, box))
        elif occ[i] < occ[i - 1]:
            after = [d.box for j in range(i, min(n, i + look + 2)) for d in people[j]]
            cands = [d for j in range(max(0, i - look - 1), i) for d in people[j]
                     if all(iou(d.box, b) < 0.3 for b in after)]
            box = max(cands, key=lambda d: d.score).box if cands else None
            exits.append((i, box))
    return entries, exits


def occupancy(tracks: list[Track], n: int, max_fill: int = 6) -> np.ndarray:
    """Number of in-room people per sample; short detection gaps inside a track are filled."""
    occ = np.zeros(n, dtype=int)
    for t in tracks:
        keys = sorted(t.boxes)
        present = np.zeros(n, dtype=bool)
        for a, b in zip(keys, keys[1:] + [keys[-1]]):
            present[a] = True
            if b - a <= max_fill:
                present[a:b + 1] = True
        occ += present
    return occ


def median_box(boxes) -> tuple | None:
    if not boxes:
        return None
    arr = np.array(boxes, dtype=np.float32)
    return tuple(float(v) for v in np.median(arr, axis=0))


@dataclass
class Layout:
    desk: tuple | None
    laptop: tuple | None
    safe: tuple | None
    door: tuple | None          # where people enter/leave the room
    machine: tuple | None


def estimate_layout(clip: VideoClip, coco: list[list[Det]], owl: dict[int, list[Det]], tracks: list[Track]) -> Layout:
    H, W = clip.height, clip.width
    laptops = [d.box for f in coco for d in f if d.label == "laptop" and d.score >= 0.6]
    laptop = median_box(laptops)

    def best_owl(label, min_score, pick=None):
        cands = [d for f in owl.values() for d in f if d.label == label and d.score >= min_score]
        if pick:
            cands = [d for d in cands if pick(d)]
        if not cands:
            return None
        # cluster around the best-scoring box, take the median of consistent boxes
        best = max(cands, key=lambda d: d.score)
        cluster = [d.box for d in cands if iou(d.box, best.box) > 0.4]
        return median_box(cluster)

    def desk_pick(d: Det) -> bool:
        # the cashier desk holds the laptop; the big white table in the foreground touches the bottom edge
        if laptop is not None:
            lx, ly = (laptop[0] + laptop[2]) / 2, (laptop[1] + laptop[3]) / 2
            return d.box[0] - 10 <= lx <= d.box[2] + 10 and d.box[1] - 40 <= ly <= d.box[3] + 10
        return d.box[3] < 0.95 * H

    desk = best_owl("desk", 0.3, desk_pick) or best_owl("desk", 0.3)
    safe = best_owl("safe", 0.15)
    machine = best_owl("machine", 0.12)

    # door: boxes where in-room tracks appear (after the start) or disappear (before the end)
    n = len(clip.samples)
    pts = []
    for t in tracks:
        if t.first > 2:
            pts.append(t.boxes[t.first])
        if t.last < n - 3:
            pts.append(t.boxes[t.last])
    door = median_box(pts) if pts else best_owl("door", 0.25)
    return Layout(desk=desk, laptop=laptop, safe=safe, door=door, machine=machine)


def region_diff_series(clip: VideoClip, box, person_boxes_per_sample: list[list[tuple]], ref_idx: list[int]) -> np.ndarray:
    """Mean absolute grey-level difference of a region vs. a reference (median of ref frames),
    ignoring pixels covered by people. Returns NaN where the region is mostly occluded."""
    small_h, small_w = clip.samples[0].small.shape
    sx = small_w / clip.width
    x0, y0, x1, y1 = [int(round(v * sx)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(small_w, x1), min(small_h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.full(len(clip.samples), np.nan)
    ref = np.median(np.stack([clip.samples[i].small[y0:y1, x0:x1] for i in ref_idx]).astype(np.float32), axis=0)
    out = np.full(len(clip.samples), np.nan)
    for i, s in enumerate(clip.samples):
        mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
        for pb in person_boxes_per_sample[i]:
            px0, py0, px1, py1 = [int(round(v * sx)) for v in pb]
            mask[max(0, py0 - y0 - 2):max(0, py1 - y0 + 2), max(0, px0 - x0 - 2):max(0, px1 - x0 + 2)] = False
        if mask.mean() < 0.35:
            continue
        diff = np.abs(s.small[y0:y1, x0:x1].astype(np.float32) - ref)
        out[i] = float(diff[mask].mean())
    return out


def brightness_series(clip: VideoClip) -> np.ndarray:
    return np.array([float(s.small.mean()) for s in clip.samples])


def screen_whiteness(clip: VideoClip, i: int, laptop_box) -> tuple[float, float]:
    """(mean brightness, share of bright low-saturation pixels) of the upper part of the laptop box."""
    x0, y0, x1, y1 = [int(v) for v in laptop_box]
    y1 = y0 + int((y1 - y0) * 0.6)
    img = clip.bgr(i)[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if img.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    white = ((v > 170) & (s < 50)).mean()
    return float(v.mean()), float(white)


def boxes_near(a, b, margin: float) -> bool:
    return not (a[2] < b[0] - margin or b[2] < a[0] - margin or a[3] < b[1] - margin or b[3] < a[1] - margin)


def on_desk(box, desk, frac: float = 0.5) -> bool:
    return desk is not None and overlap_frac(box, desk) >= frac
