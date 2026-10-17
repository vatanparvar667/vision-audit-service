"""End-to-end analysis of one video: perception -> scene timeline -> rules -> VLM verification -> report."""

import logging
import threading
import time
from pathlib import Path

import numpy as np

from app.config import settings
from app.pipeline import rules as R
from app.pipeline.perception import Perception
from app.pipeline.report import build_report
from app.pipeline.scene import (brightness_series, estimate_layout, in_room_tracks, occupancy_changes, person_scale,
                                region_diff_series, rolling_median, room_people, screen_whiteness, track_people)
from app.pipeline.video import VideoClip, read_video

log = logging.getLogger("analyze")

COCO_FPS = 4.0
OWL_EVERY_S = 1.0
USE_DESK_PASS = False  # second OWLv2 pass on the desk crop: no gain on the dataset, one extra false positive
SAFE_EVERY_S = 2.0
VLM_RETRY_S = 300.0  # after a failed VLM load, try again on a later job instead of never

_models: dict = {}
_models_lock = threading.Lock()


def get_perception() -> Perception:
    with _models_lock:
        if "perception" not in _models:
            _models["perception"] = Perception(settings.models_dir, settings.device)
        return _models["perception"]


def get_vlm():
    if not settings.use_vlm:
        return None
    with _models_lock:
        if _models.get("vlm") is not None:
            return _models["vlm"]
        if time.time() < _models.get("vlm_retry_at", 0.0):
            return None
        from app.pipeline.vlm import YesNoVLM

        path = settings.models_dir / "Qwen3-VL-4B-Instruct"
        try:
            if not path.exists():
                raise FileNotFoundError(f"VLM folder {path} not found")
            _models["vlm"] = YesNoVLM(path, settings.device)
        except Exception as exc:  # e.g. not enough GPU memory: rules still work, unverified
            log.error("VLM not loaded, continuing without verification (retry in %.0fs): %s", VLM_RETRY_S, exc)
            _models["vlm_retry_at"] = time.time() + VLM_RETRY_S
            return None
        return _models["vlm"]


def warmup() -> None:
    t0 = time.time()
    get_perception()
    get_vlm()
    log.info("models ready in %.1fs", time.time() - t0)


def perceive(clip: VideoClip, progress=None) -> dict:
    per = get_perception()
    coco = per.detect_coco(clip, progress=(lambda f: progress(0.15 + 0.35 * f, "detecting")) if progress else None)
    step = max(1, int(round(clip.sample_fps * OWL_EVERY_S)))
    idx = list(range(0, len(clip.samples), step))
    owl = per.detect_owl(clip, idx, progress=(lambda f: progress(0.5 + 0.1 * f, "detecting")) if progress else None)
    if USE_DESK_PASS:
        owl = add_desk_pass(clip, coco, owl, idx, progress)
    return {"coco": coco, "owl": owl}


def add_desk_pass(clip: VideoClip, coco, owl, idx, progress=None):
    """Second OWLv2 pass on the enlarged desk region: banknote bundles are only ~15-30 px in the full frame."""
    from app.pipeline.perception import desk_crop_box, merge_owl

    desk = estimate_layout(clip, coco, owl, []).desk
    crop = desk_crop_box(desk, clip.width, clip.height)
    if crop is None:
        return owl
    extra = get_perception().detect_owl(clip, idx, crop=crop,
                                        progress=(lambda f: progress(0.6 + 0.1 * f, "detecting")) if progress else None)
    return merge_owl(owl, extra)


def build_timeline(clip: VideoClip, coco, owl) -> R.Timeline:
    all_tracks = track_people(coco, clip.height)
    scale = person_scale(all_tracks, clip.width, clip.height)
    tracks = in_room_tracks(all_tracks, clip.width, clip.height, scale=scale)
    people = [room_people(f, clip.width, clip.height, scale) for f in coco]
    occ = rolling_median(np.array([len(p) for p in people]), 7)
    entries, exits = occupancy_changes(people, occ)
    layout = estimate_layout(clip, coco, owl, tracks)
    tl = R.Timeline(t=np.array([s.t for s in clip.samples]), occ=occ, tracks=tracks, coco=coco, owl=owl,
                    layout=layout, width=clip.width, height=clip.height, duration=clip.duration,
                    people=people, entries=entries, exits=exits)
    tl.brightness = brightness_series(clip)
    if layout.laptop is not None:
        tl.screen = {i: screen_whiteness(clip, i, layout.laptop) for i in range(0, len(clip.samples), 2)}
    return tl


def door_state(clip: VideoClip, tl: R.Timeline) -> tuple[dict | None, np.ndarray | None]:
    """Compare the end of the video with its start near the paths people used to enter/leave."""
    from app.pipeline.doors import door_change
    from app.pipeline.perception import iou

    fps = R._fps(tl)
    n = len(tl.t)
    events = sorted([k for k, _ in tl.entries] + [k for k, _ in tl.exits])
    if not events or n < int(8 * fps):
        return None, None
    if tl.brightness is not None and abs(float(tl.brightness[-1]) - float(tl.brightness[0])) > 25:
        return None, None  # lights switched on/off: pixel comparison is meaningless
    path = []
    for k in events:
        before = [p.box for p in tl.people[max(0, k - int(3 * fps))]]
        path += [d.box for i in range(max(0, k - int(2.5 * fps)), min(n, k + int(1 * fps)))
                 for d in tl.coco[i] if d.label == "person" and d.score >= 0.3 and all(iou(d.box, x) < 0.3 for x in before)]
    ref = list(range(0, min(3, n)))
    k_last = n - int(3.5 * fps) - int(2.5 * fps)
    blob = door_change(clip, tl.coco, k_last, fps, path, tl.layout.desk, ref_idx=ref, post_end=n)
    if not blob or not blob.get("box"):
        return blob, None
    persons = [[d.box for d in f if d.label == "person" and d.score >= 0.3] for f in tl.coco]
    return blob, region_diff_series(clip, blob["box"], persons, ref)


def run_rules(clip: VideoClip, tl: R.Timeline, safe_p: dict[int, float]) -> list[R.Event]:
    events = []
    events += R.rule_left_unattended(tl)
    events += R.rule_food_drink(tl)
    events += R.rule_personal_items(tl)
    door_events = R.rule_door_open(tl, *door_state(clip, tl))
    events += door_events
    events += R.rule_safe_open(tl, safe_p)
    events += R.rule_client_cash(tl)
    events += R.rule_second_visitor(tl)
    events += R.rule_bag_handover(tl)
    events += R.rule_outsider(tl, door_events)
    return sorted(events, key=lambda e: e.t)


def safe_probs(clip: VideoClip, vlm, progress=None) -> dict[int, float]:
    if vlm is None:
        return {}
    step = max(1, int(round(clip.sample_fps * SAFE_EVERY_S)))
    idx = list(range(0, len(clip.samples), step))
    out = {}
    for k, i in enumerate(idx):
        try:
            out[i] = vlm.yes_prob(clip.rgb(i), "Is the door of the metal safe open?")
        except Exception as exc:  # a VLM failure must not fail the whole video
            log.error("safe check failed at sample %d, skipping the rest: %s", i, exc)
            _free_gpu()
            break
        if progress:
            progress(0.7 + 0.15 * (k + 1) / len(idx), "verifying")
    return out


def _free_gpu() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def verify(clip: VideoClip, events: list[R.Event], vlm) -> list[R.Event]:
    """Ask the VLM the event's own question(s); drop events it clearly rejects."""
    if vlm is None:
        return events
    from app.pipeline.vlm import crop

    kept = []
    for e in events:
        probs = []
        try:
            for t, box, question in e.verify:
                img = clip.rgb(clip.nearest(t))
                probs.append(vlm.yes_prob(crop(img, box) if box is not None else img, question))
        except Exception as exc:  # keep the rule's own score when the verifier fails
            log.error("verification of %s failed: %s", e.type, exc)
            _free_gpu()
            probs = []
        e.facts["vlm"] = [round(p, 3) for p in probs]
        if probs:
            p = float(np.mean(probs))
            e.score = 0.5 * e.score + 0.5 * p
            if p < 0.2:
                continue
        kept.append(e)
    return kept


def analyze_clip(clip: VideoClip, perception: dict | None = None, progress=None, use_vlm: bool = True) -> dict:
    perception = perception or perceive(clip, progress)
    tl = build_timeline(clip, perception["coco"], perception["owl"])
    vlm = get_vlm() if use_vlm else None
    sp = safe_probs(clip, vlm, progress)
    events = run_rules(clip, tl, sp)
    if progress:
        progress(0.88, "verifying")
    events = verify(clip, events, vlm)
    if progress:
        progress(0.95, "report")
    report = build_report(clip, tl, events)
    if use_vlm and settings.use_vlm and vlm is None:
        report["notes"] = ("Проверка через визуально-языковую модель недоступна: открытый сейф, напитки и личные "
                           "вещи в этом прогоне не проверялись.")
    return report


def analyze_video(path: Path, progress=None) -> dict:
    if progress:
        progress(0.02, "decoding")
    clip = read_video(str(path), sample_fps=COCO_FPS,
                      progress=(lambda f: progress(0.02 + 0.13 * f, "decoding")) if progress else None)
    return analyze_clip(clip, progress=progress)
