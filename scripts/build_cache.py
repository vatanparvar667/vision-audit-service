"""Run perception on dataset videos once and cache results (for fast offline rule tuning).
Usage: python -I scripts/build_cache.py VIDEO_DIR OUT_DIR [names...]"""
import pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.pipeline.video import read_video
from app.pipeline.perception import Perception

vdir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
names = sys.argv[3:] or sorted(p.stem for p in vdir.glob("*.mov"))
per = Perception(settings.models_dir)
for name in names:
    dst = out / f"{name}.pkl"
    if dst.exists(): continue
    t0 = time.time()
    clip = read_video(vdir / f"{name}.mov", sample_fps=4.0)
    t1 = time.time()
    coco = per.detect_coco(clip)
    t2 = time.time()
    step = max(1, int(round(clip.sample_fps)))  # ~1 fps for OWLv2
    owl = per.detect_owl(clip, list(range(0, len(clip.samples), step)))
    t3 = time.time()
    pickle.dump({"clip": clip, "coco": coco, "owl": owl}, dst.open("wb"))
    print(f"{name}: dur {clip.duration:.1f}s samples {len(clip.samples)} read {t1-t0:.1f}s coco {t2-t1:.1f}s owl {t3-t2:.1f}s", flush=True)
