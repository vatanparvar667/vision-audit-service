"""Temporal rules over the scene timeline. Each rule yields candidate events; the VLM verifier and
the report builder decide what is shown to the user."""

from dataclasses import dataclass, field

import numpy as np

from app.pipeline.perception import Det, iou, overlap_frac
from app.pipeline.scene import Layout, Track, on_desk


@dataclass
class Event:
    type: str
    t: float                    # timecode reported to the user (seconds)
    t_start: float
    t_end: float
    score: float                # rule confidence 0..1 before verification
    evidence_t: list[float]     # 1-3 timestamps for evidence frames
    boxes: dict = field(default_factory=dict)   # evidence_t -> list of (label, box) to highlight
    facts: dict = field(default_factory=dict)   # observed values used in the description
    verify: list = field(default_factory=list)  # VLM checks: (t, box or None, question)


@dataclass
class Timeline:
    """Everything rules need, indexed by sample."""
    t: np.ndarray                       # sample times
    occ: np.ndarray                     # in-room people per sample
    tracks: list[Track]
    coco: list[list[Det]]
    owl: dict[int, list[Det]]
    layout: Layout
    width: int
    height: int
    duration: float
    door_diff: np.ndarray | None = None
    brightness: np.ndarray | None = None
    screen: dict[int, tuple[float, float]] | None = None   # sample -> (brightness, white share)
    people: list[list[Det]] = field(default_factory=list)  # in-room person detections per sample
    entries: list = field(default_factory=list)             # (sample, box) where the head count goes up
    exits: list = field(default_factory=list)               # (sample, box) where it goes down

    def idx(self, t: float) -> int:
        return int(np.argmin(np.abs(self.t - t)))

    def owl_near(self, i: int, max_gap: int = 4) -> tuple[int, list[Det]] | None:
        keys = sorted(self.owl)
        if not keys:
            return None
        j = min(keys, key=lambda k: abs(k - i))
        return (j, self.owl[j]) if abs(j - i) <= max_gap else None


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [a, b] where mask is True."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def smooth_occ(occ: np.ndarray, k: int = 5) -> np.ndarray:
    pad = k // 2
    p = np.pad(occ, pad, mode="edge")
    return np.array([int(np.median(p[i:i + k])) for i in range(len(occ))])


# ---------------------------------------------------------------------------------------------
# Rule: cashier leaves the room/workplace -> what is left behind (cash, bag, documents, open screen)
# ---------------------------------------------------------------------------------------------

CASH_MIN = 0.22      # OWLv2 score for banknotes on the desk
BAG_MIN = 0.30
DOC_MIN = 0.28


def desk_items(tl: Timeline, i: int, labels=("cash", "bag", "document")) -> list[Det]:
    got = tl.owl_near(i)
    if not got:
        return []
    _, dets = got
    mins = {"cash": CASH_MIN, "bag": BAG_MIN, "document": DOC_MIN}
    return [d for d in dets if d.label in labels and d.score >= mins[d.label] and on_desk(d.box, tl.layout.desk, 0.6)]


ITEM_Q = {
    "cash": "Is there a stack of banknotes (paper money) in this picture?",
    "bag": "Is there a bag (a fabric bag, a cash bag or a handbag) in this picture?",
    "document": "Are there documents, a folder or a booklet in this picture?",
}


def rule_left_unattended(tl: Timeline) -> list[Event]:
    events = []
    empty = tl.occ == 0
    label_ru = {"cash": "наличные", "bag": "сумка", "document": "документы", "item": "посторонние предметы"}
    merged = []
    for a, b in runs(empty):
        if merged and tl.t[a] - tl.t[merged[-1][1]] <= 1.2:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    # only "leaving" counts: the room must have been occupied before
    for a, b in merged:
        if a == 0 or tl.t[b] - tl.t[a] < 0.7:
            continue
        t_exit = float(tl.t[a])
        t_ev = max(0.0, t_exit - 4.0)  # head count drops ~2-3 s after the person walks out
        i_check = min(b, a + 2)
        found: dict[str, Det] = {}
        seen_at = i_check
        for i in range(a, b + 1):
            cands = desk_items(tl, i) if i in tl.owl else []
            # COCO sees closed booklets as "book" and cash capsules / coin tubes as "cup" or "bottle"
            cands += [Det("document", d.score, d.box) for d in tl.coco[i]
                      if d.label == "book" and d.score >= 0.42 and on_desk(d.box, tl.layout.desk, 0.6)]
            cands += [Det("item", d.score, d.box) for d in tl.coco[i]
                      if d.label in ("cup", "bottle") and d.score >= 0.35 and on_desk(d.box, tl.layout.desk, 0.6)]
            for d in cands:
                if d.label not in found or d.score > found[d.label].score:
                    found[d.label] = d
                    seen_at = i
        if not found:
            # weaker evidence: an object seen on the desk just before the exit stays at the same place
            # while the room is empty (OWLv2 may change its label between frames: bag -> document)
            fps = _fps(tl)
            before = [d for i in range(max(0, a - int(3 * fps)), a) if i in tl.owl
                      for d in desk_items(tl, i)]
            during = [(i, d) for i in range(a, b + 1) if i in tl.owl for d in tl.owl[i]
                      if d.label in ("cash", "bag", "document") and d.score >= 0.2 and on_desk(d.box, tl.layout.desk, 0.6)]
            for p in sorted(before, key=lambda d: -d.score):
                same = [(i, d) for i, d in during if iou(p.box, d.box) > 0.5]
                if same:
                    found[p.label] = p
                    seen_at = same[0][0]
                    break
        kinds = sorted(found)
        # always ask the verifier about the desk: closed booklets, cash bags etc. are often missed by OWLv2
        events.append(Event(
            type="cash_unattended", t=t_ev, t_start=t_ev, t_end=float(tl.t[b]),
            score=(0.5 + 0.5 * max(d.score for d in found.values())) if found else 0.42,
            evidence_t=[t_ev, float(tl.t[seen_at])],
            boxes={float(tl.t[seen_at]): ([(label_ru[k], found[k].box) for k in kinds] or [("стол", tl.layout.desk)])},
            facts={"items": [label_ru[k] for k in kinds] or ["предметы, оставленные на столе"], "t_exit": t_exit,
                   "empty_for": float(tl.t[b] - tl.t[a])},
            # items found by OWLv2 are trusted (the VLM misreads banknote bundles at this resolution);
            # without them the VLM is asked about the whole desk
            verify=([] if set(found) - {"item"} else
                    [(float(tl.t[seen_at]), tl.layout.desk,
                      "Nobody is at this desk. Are banknotes, a bag, documents, a booklet or other valuables left "
                      "lying on it (ignore the computer, the desk phone and the money counting machine)?")]),
        ))
        # screen left unlocked: the screen stays bright relative to the room
        if tl.screen and tl.layout.laptop is not None and tl.brightness is not None:
            ids = [i for i in range(a, b + 1) if i in tl.screen]
            if ids:
                sv = float(np.median([tl.screen[i][0] for i in ids]))
                gm = float(np.median([tl.brightness[i] for i in ids]))
                ratio = sv / max(1.0, gm)
                if ratio >= 1.05:
                    i_mid = ids[len(ids) // 2]
                    events.append(Event(
                        type="pc_unlocked", t=t_ev, t_start=t_ev, t_end=float(tl.t[b]),
                        score=min(0.85, 0.5 + (ratio - 1.0)),
                        evidence_t=[t_ev, float(tl.t[i_mid])],
                        boxes={float(tl.t[i_mid]): [("экран", tl.layout.laptop)]},
                        facts={"t_exit": t_exit, "ratio": ratio},
                        verify=[(float(tl.t[i_mid]), tl.layout.laptop,
                                 "Is this laptop screen switched on and showing a window or application "
                                 "(not a lock screen with a clock, not black)?")],
                    ))
    return events


# ---------------------------------------------------------------------------------------------
# Rule: drinking / eating at the desk (cup or bottle at the face of a person)
# ---------------------------------------------------------------------------------------------

def rule_food_drink(tl: Timeline) -> list[Event]:
    # a real cup/bottle must be clearly visible at some point (banknotes or cards at the face are not drinks)
    best_cup = max((d.score for f in tl.coco for d in f if d.label in ("cup", "bottle", "wine glass")), default=0.0)
    if best_cup < 0.6:
        return []
    hits = []
    for i, frame in enumerate(tl.coco):
        cups = [d for d in frame if d.label in ("cup", "bottle", "wine glass") and d.score >= 0.35]
        if not cups:
            continue
        for person in tl.people[i]:
            x0, y0, x1, y1 = person.box
            face_y = y0 + 0.45 * (y1 - y0)
            for c in cups:
                if x0 - 5 <= c.cx <= x1 + 5 and y0 - 10 <= c.cy <= face_y:
                    hits.append((i, c, person.box))
    if not hits:
        return []
    events = []
    idxs = sorted({h[0] for h in hits})
    # group hits closer than 3 s
    groups, cur = [], [idxs[0]]
    for i in idxs[1:]:
        if tl.t[i] - tl.t[cur[-1]] <= 3.0:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    for g in groups:
        if len(g) < 2:
            continue
        best = max((h for h in hits if h[0] in g), key=lambda h: h[1].score)
        i, cup, pbox = best
        t0 = float(tl.t[g[0]])
        # the episode starts when the cup is first in the person's hands, a few seconds before it reaches the face
        for j in range(g[0] - 1, max(-1, g[0] - int(6 * _fps(tl))), -1):
            near = [c for c in tl.coco[j] if c.label in ("cup", "bottle", "wine glass") and c.score >= 0.35
                    and pbox[0] - 10 <= c.cx <= pbox[2] + 10 and pbox[1] - 10 <= c.cy <= pbox[3] + 10]
            if not near:
                break
            t0 = float(tl.t[j])
        events.append(Event(
            type="food_drink", t=t0, t_start=t0, t_end=float(tl.t[g[-1]]), score=0.45 + 0.1 * cup.score,
            evidence_t=[float(tl.t[i])], boxes={float(tl.t[i]): [("напиток", cup.box)]},
            facts={"duration": float(tl.t[g[-1]] - tl.t[g[0]])},
            verify=[(float(tl.t[i]), pbox, "Is this person drinking or eating something?")],
        ))
    return events


# ---------------------------------------------------------------------------------------------
# Rule: personal bag / clothes on the cash desk
# ---------------------------------------------------------------------------------------------

def rule_personal_items(tl: Timeline) -> list[Event]:
    if tl.layout.desk is None:
        return []
    hits = []
    for i, dets in tl.owl.items():
        for d in dets:
            if d.label in ("bag", "clothes") and d.score >= (0.45 if d.label == "bag" else 0.4) and on_desk(d.box, tl.layout.desk, 0.6):
                if tl.occ[i] >= 1:
                    hits.append((i, d))
    for i, frame in enumerate(tl.coco):
        for d in frame:
            if d.label in ("handbag", "backpack") and d.score >= 0.7 and on_desk(d.box, tl.layout.desk, 0.6):
                hits.append((i, d))
    if len(hits) < 3:
        return []
    hits.sort(key=lambda h: h[0])
    span = tl.t[hits[-1][0]] - tl.t[hits[0][0]]
    if span < 2.0:
        return []
    i, d = max(hits, key=lambda h: h[1].score)
    t0 = float(tl.t[hits[0][0]])
    return [Event(
        type="personal_items", t=t0, t_start=t0, t_end=float(tl.t[hits[-1][0]]), score=0.45,  # needs the VLM to pass
        evidence_t=[float(tl.t[i])], boxes={float(tl.t[i]): [("личная вещь", d.box)]},
        facts={"span": float(span)},
        verify=[(float(tl.t[i]), tl.layout.desk,
                 "Is there a personal handbag, a purse or a piece of clothing (jacket, coat) lying on this desk? "
                 "A green cash-collection bag does not count.")],
    )]


# ---------------------------------------------------------------------------------------------
# Entries / exits of people (track births and deaths that are not tracker fragmentation)
# ---------------------------------------------------------------------------------------------

def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def cashier_track(tl: Timeline) -> Track | None:
    return max(tl.tracks, key=lambda t: len(t.boxes), default=None)


def _fps(tl: Timeline) -> float:
    return 1.0 / max(1e-6, float(np.median(np.diff(tl.t)))) if len(tl.t) > 1 else 4.0


def _dedup(events: list[Event], window: float) -> list[Event]:
    out = []
    for e in sorted(events, key=lambda e: -e.score):
        if all(o.type != e.type or abs(o.t - e.t) > window for o in out):
            out.append(e)
    return sorted(out, key=lambda e: e.t)


# ---------------------------------------------------------------------------------------------
# Rule: room door left open after somebody entered or left
# ---------------------------------------------------------------------------------------------

DOOR_BLOB_MIN = 0.010      # share of frame that must stay changed near the entry path
DOOR_DIFF_OPEN = 12.0


def rule_door_open(tl: Timeline, blob: dict | None, series: np.ndarray | None) -> list[Event]:
    """blob: persistent change near the entry path at the end of the video vs. its start (doors.door_change);
    series: grey-level difference of that blob region over time (people masked)."""
    if not blob or not blob.get("box") or blob["area"] < DOOR_BLOB_MIN or series is None:
        return []
    x0, y0, x1, y1 = blob["box"]
    cy = (y0 + y1) / 2
    limit = (tl.layout.desk[1] - 0.08 * tl.height) if tl.layout.desk is not None else 0.45 * tl.height
    if cy > limit:
        return []  # changes low in the frame are furniture (chairs), not the door
    # earliest moment after which the region stays changed until the end
    valid = [(i, v) for i, v in enumerate(series) if not np.isnan(v)]
    if len(valid) < 6:
        return []
    t_open = None
    for j in range(len(valid)):
        tail = np.array([v for _, v in valid[j:]])
        if tail[0] >= DOOR_DIFF_OPEN and len(tail) >= 4 and (tail >= DOOR_DIFF_OPEN).mean() >= 0.85:
            t_open = float(tl.t[valid[j][0]])
            break
    if t_open is None or tl.duration - t_open < 4.0:
        return []
    # the start of the clip is the "closed" reference: it must really show the door region clear of people,
    # and the region must not differ already at the start (then the door may have been open and got closed)
    if t_open < 1.0:
        return []
    for i in range(min(3, len(tl.t))):
        for p in tl.coco[i]:
            if p.label == "person" and p.score >= 0.3 and max(overlap_frac(blob["box"], p.box),
                                                               overlap_frac(p.box, blob["box"])) > 0.3:
                return []
    i_ev = tl.idx(t_open)
    return [Event(
        type="door_open", t=t_open, t_start=t_open, t_end=tl.duration,
        score=min(0.9, 0.55 + blob["area"] * 6),
        evidence_t=[t_open, float(tl.t[min(len(tl.t) - 1, i_ev + int(4 * _fps(tl)))])],
        boxes={t_open: [("дверь", blob["box"])]},
        facts={"open_for": float(tl.duration - t_open), "t_entry": t_open},
    )]


# ---------------------------------------------------------------------------------------------
# Rule: safe open while the cashier sits at the desk or is away
# ---------------------------------------------------------------------------------------------

def rule_safe_open(tl: Timeline, safe_p: dict[int, float]) -> list[Event]:
    if not safe_p:
        return []
    keys = sorted(safe_p)
    cash = cashier_track(tl)
    hmax = max((b[3] - b[1] for b in cash.boxes.values()), default=1.0) if cash else 1.0
    open_mask = np.array([safe_p[k] >= 0.7 for k in keys])
    events = []
    for a, b in runs(open_mask):
        ka, kb = keys[a], keys[b]
        if tl.t[kb] - tl.t[ka] < 4.0:
            continue
        # cashier away from the safe: room empty, or she opened it standing and then sat down at the desk.
        # A cashier who works at a safe within reach of her chair is seated the whole time: not a violation.
        h_open = None
        if cash:
            near_open = [k for k in sorted(cash.boxes) if abs(k - ka) <= int(2 * _fps(tl))]
            if near_open:
                b0 = cash.boxes[min(near_open, key=lambda k: abs(k - ka))]
                h_open = b0[3] - b0[1]
        away = []
        for k in keys[a:b + 1]:
            if tl.occ[k] == 0:
                away.append(k)
            elif cash and k in cash.boxes and h_open is not None:
                h = cash.boxes[k][3] - cash.boxes[k][1]
                if h < 0.75 * h_open and h < 0.8 * hmax:
                    away.append(k)
        if len(away) < 2 or tl.t[away[-1]] - tl.t[away[0]] < 2.5:
            continue
        t_ev = float(tl.t[away[0]])
        mid = away[len(away) // 2]
        events.append(Event(
            type="safe_open", t=t_ev, t_start=float(tl.t[ka]), t_end=float(tl.t[kb]),
            score=float(np.mean([safe_p[k] for k in keys[a:b + 1]])),
            evidence_t=[t_ev, float(tl.t[mid])],
            boxes={t_ev: [("сейф", tl.layout.safe)]} if tl.layout.safe else {},
            facts={"open_for": float(tl.t[kb] - tl.t[ka]), "away_for": float(tl.t[away[-1]] - tl.t[away[0]]),
                   "empty_room": any(tl.occ[k] == 0 for k in away)},
            verify=[(float(tl.t[mid]), None, "Is the metal safe standing open while nobody is standing at it "
                                             "(the person is sitting at the desk or away from the safe)?")],
        ))
    return events


# ---------------------------------------------------------------------------------------------
# Rules around clients: cash left after the client leaves, second visitor, outsider, bag hand-over
# ---------------------------------------------------------------------------------------------

def _grow(box, k: float):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * k / 2, (box[3] - box[1]) * k / 2
    return (cx - w, cy - h, cx + w, cy + h)


def cash_on_desk(tl: Timeline, i: int, min_score: float = CASH_MIN) -> list[Det]:
    got = tl.owl_near(i, max_gap=3)
    if not got:
        return []
    return [d for d in got[1] if d.label == "cash" and d.score >= min_score and on_desk(d.box, tl.layout.desk, 0.5)]


def visitor_exits(tl: Timeline) -> list[tuple[int, tuple | None]]:
    """Exits after which somebody (the cashier) is still in the room."""
    return [(k, b) for k, b in tl.exits if tl.occ[k] >= 1 and tl.occ[k - 1] >= 2]


def visitor_entries(tl: Timeline) -> list[tuple[int, tuple | None]]:
    """Entries that join somebody already in the room."""
    return [(k, b) for k, b in tl.entries if tl.occ[k - 1] >= 1 and tl.occ[k] >= 2]


def rule_client_cash(tl: Timeline) -> list[Event]:
    if tl.layout.desk is None:
        return []
    events = []
    fps = _fps(tl)
    used_next: set[int] = set()
    for k, box in visitor_exits(tl):
        if tl.occ[k] != 1:
            continue  # the client left and only the cashier remains
        # the visitor must have stayed a while (a client, not a passer-by)
        prev_entries = [e for e, _ in tl.entries if e < k]
        if prev_entries and tl.t[k] - tl.t[prev_entries[-1]] < 6.0:
            continue
        seen = []
        for i in range(max(0, k - int(1 * fps)), min(len(tl.t), k + int(6 * fps))):
            if i in tl.owl:
                seen += [(i, d) for d in cash_on_desk(tl, i)]
        if not seen:
            continue
        i_best, d_best = max(seen, key=lambda s: s[1].score)
        t_exit = float(tl.t[k])
        nxt = [e for e, _ in visitor_entries(tl) if e > k and tl.t[e] - t_exit <= 30 and tl.occ[e - 1] == 1]
        if nxt:
            e = nxt[0]
            if e in used_next:
                continue  # an earlier exit already reported the cash seen at this entry
            still = [d for i in range(max(k, e - int(3 * fps)), min(len(tl.t), e + int(2 * fps)))
                     if i in tl.owl for d in cash_on_desk(tl, i)]
            if still:
                used_next.add(e)
                t_ev = float(tl.t[e])
                events.append(Event(
                    type="prev_client_cash", t=t_ev, t_start=t_exit, t_end=t_ev,
                    score=min(0.9, 0.45 + max(d.score for d in still)),
                    evidence_t=[float(tl.t[i_best]), t_ev],
                    boxes={float(tl.t[i_best]): [("наличные", d_best.box)],
                           t_ev: [("наличные", max(still, key=lambda d: d.score).box)]},
                    facts={"t_exit": t_exit, "t_next": t_ev},
                    verify=[],  # OWLv2 banknote detections are persistent over seconds; no VLM veto
                ))
                continue
        events.append(Event(
            type="count_without_client", t=t_exit, t_start=t_exit, t_end=float(tl.t[i_best]),
            score=min(0.85, 0.35 + d_best.score),
            evidence_t=[t_exit, float(tl.t[i_best])],
            boxes={float(tl.t[i_best]): [("наличные", d_best.box)]},
            facts={"t_exit": t_exit},
            verify=[],
        ))
    return events


def rule_second_visitor(tl: Timeline) -> list[Event]:
    events = []
    spans = []
    for a, b in runs(tl.occ >= 3):
        if tl.t[b] - tl.t[a] < 4.0:
            continue
        if spans and tl.t[a] - tl.t[spans[-1][1]] <= 3.0:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    for a, b in spans:
        # report the moment the extra person is already at the desk, not the moment they walk in
        t_ev = float(tl.t[a]) + min(4.0, float(tl.t[b] - tl.t[a]) / 2)
        mid = (a + b) // 2
        events.append(Event(
            type="second_visitor", t=t_ev, t_start=float(tl.t[a]), t_end=float(tl.t[b]), score=0.6,
            evidence_t=[t_ev, float(tl.t[mid])],
            boxes={t_ev: [("человек", d.box) for d in tl.people[tl.idx(t_ev)]]},
            facts={"people": int(tl.occ[a:b + 1].max()), "duration": float(tl.t[b] - tl.t[a])},
            verify=[(float(tl.t[mid]), None, "Are there at least three people inside this room (not counting people "
                                             "seen through the glass wall or reflections)?")],
        ))
    return events


def _bag_dets(tl: Timeline, i: int) -> list[Det]:
    out = [d for d in tl.coco[i] if d.label in ("handbag", "backpack", "suitcase") and d.score >= 0.4]
    if i in tl.owl:
        out += [d for d in tl.owl[i] if d.label == "bag" and d.score >= 0.3]
    return out


def rule_bag_handover(tl: Timeline) -> list[Event]:
    """A visitor comes in, leaves a bag on the desk and is gone within seconds."""
    if tl.layout.desk is None:
        return []
    events = []
    fps = _fps(tl)
    exits = visitor_exits(tl)
    for e, _ in visitor_entries(tl):
        if tl.occ[e - 1] != 1:
            continue
        later = [k for k, _ in exits if k > e]
        if not later or tl.occ[later[0]] != 1:
            continue
        k = later[0]
        stay = float(tl.t[k] - tl.t[e])
        if stay > 12.0 or stay < 2.0:
            continue
        before = [d.box for j in range(max(0, e - int(10 * fps)), e) for d in _bag_dets(tl, j)]
        bags = [(i, d) for i in range(e, k + 1) for d in _bag_dets(tl, i)
                if on_desk(d.box, tl.layout.desk, 0.5) and all(iou(d.box, bb) < 0.3 for bb in before)]
        if len(bags) < 2:
            continue
        i, d = bags[0]
        t_ev = float(tl.t[i])
        events.append(Event(
            type="bag_no_check", t=t_ev, t_start=float(tl.t[e]), t_end=float(tl.t[k]),
            score=0.55, evidence_t=[t_ev], boxes={t_ev: [("сумка", d.box)]},
            facts={"stay": stay},
            verify=[(t_ev, None, "Is a person handing over or putting down a bag at the cashier's desk?")],
        ))
    return events


def rule_outsider(tl: Timeline, door_events: list[Event]) -> list[Event]:
    """A person joining the cashier through a door that had been left open."""
    events = []
    fps = _fps(tl)
    for e, box in visitor_entries(tl):
        te = float(tl.t[e])
        if any(d.t_start + 1.0 <= te <= d.t_end + 1.5 for d in door_events):
            stay_idx = next((k for k, _ in tl.exits if k > e), len(tl.t) - 1)
            events.append(Event(
                type="outsider", t=te, t_start=te, t_end=float(tl.t[stay_idx]), score=0.6,
                evidence_t=[te, float(tl.t[min(len(tl.t) - 1, e + int(2 * fps))])],
                boxes={te: [("вошедший", box)]} if box else {},
                facts={"stay": float(tl.t[stay_idx] - te)},
            ))
    return events
