"""Print occupancy timeline and tracks per video, to calibrate person filtering."""
import pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.scene import track_people, in_room_tracks, occupancy, estimate_layout
for name in sys.argv[1:]:
    d = pickle.load(open(f"runs/cache/{name}.pkl", "rb")); clip = d["clip"]
    tr = track_people(d["coco"], clip.height)
    room = in_room_tracks(tr, clip.width, clip.height)
    occ = occupancy(room, len(clip.samples))
    lay = estimate_layout(clip, d["coco"], d["owl"], room)
    ts = [s.t for s in clip.samples]
    print(f"== {name} dur {clip.duration:.1f} tracks {len(tr)} in-room {len(room)}")
    for t in room:
        b0 = t.boxes[t.first]; b1 = t.boxes[t.last]
        print(f"   T{t.tid}: {ts[t.first]:.1f}-{ts[t.last]:.1f}s n={len(t.boxes)} peak={t.peak():.2f} med={t.median_score():.2f} start=({b0[0]:.0f},{b0[1]:.0f},{b0[2]:.0f},{b0[3]:.0f}) end=({b1[0]:.0f},{b1[1]:.0f},{b1[2]:.0f},{b1[3]:.0f})")
    # compress occupancy
    segs = []; cur = None
    for i, o in enumerate(occ):
        if cur is None or o != cur[0]:
            cur = [o, ts[i], ts[i]]; segs.append(cur)
        else: cur[2] = ts[i]
    print("   occ:", " ".join(f"{o}@{a:.1f}-{b:.1f}" for o, a, b in segs))
    print("   layout:", {k: tuple(round(v) for v in val) if val else None for k, val in lay.__dict__.items()})
