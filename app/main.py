"""FastAPI entry point: web UI, the jury API (/api/v1, X-API-Key) and keyless UI endpoints (/ui)."""

import hmac
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.jobs import DONE, FAILED, JobStore, Worker
from app.norms import catalog_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

store = JobStore(settings.storage_dir)


def _process(job_id: str, video_path: Path, progress) -> dict:
    from app.pipeline.analyze import analyze_video  # heavy imports stay out of web startup

    return analyze_video(video_path, progress=progress)


worker = Worker(store, _process, keep_videos=settings.keep_videos)


def _warmup() -> None:
    from app.pipeline.analyze import warmup

    warmup()


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker.start()
    # load the models in the background so the first uploaded video does not pay for it
    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield


app = FastAPI(title="ForteBank cash-desk video audit", version="1.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in settings.api_keys):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def public_view(job: dict) -> dict:
    """Shape a job record into the response schema from the case (section 10)."""
    out = {"job_id": job["job_id"], "status": job["status"]}
    if job["status"] == DONE:
        out["verdict"] = job["verdict"]
        out["violations"] = job["violations"]
        out["processing_sec"] = job.get("processing_sec")
        if job.get("notes"):
            out["notes"] = job["notes"]
    elif job["status"] == FAILED:
        out["error"] = job.get("error", "processing failed")
    else:
        out["stage"] = job.get("stage")
        out["progress"] = job.get("progress", 0.0)
        pos = store.queue_position(job["job_id"])
        if pos >= 0:
            out["queue_position"] = pos
    return out


async def _accept_upload(file: UploadFile) -> dict:
    name = file.filename or "video.mov"
    suffix = Path(name).suffix.lower() or ".mov"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"unsupported file type {suffix}; expected .mov")
    job = store.new_job(name)
    dest = store.video_path(job["job_id"], suffix)
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                store.update(job["job_id"], persist=True, status=FAILED, stage="failed", error="file too large")
                raise HTTPException(status_code=413, detail=f"file larger than {settings.max_upload_mb} MB")
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        store.update(job["job_id"], persist=True, status=FAILED, stage="failed", error="empty file")
        raise HTTPException(status_code=400, detail="empty file")
    store.update(job["job_id"], size_bytes=size)
    worker.submit(job["job_id"], dest)
    log.info("job %s queued: %s (%.1f MB)", job["job_id"], name, size / 1e6)
    return {"job_id": job["job_id"], "status": job["status"]}


def _job_or_404(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# ---- jury API -------------------------------------------------------------------------------

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def api_analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def api_job(job_id: str):
    return public_view(_job_or_404(job_id))


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "queue": worker.queue.qsize()}


@app.get("/api/v1/violation-types")
def violation_types():
    return catalog_for_api()


# ---- web UI (no auth, see FAQ 27) --------------------------------------------------------------

@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    return public_view(_job_or_404(job_id))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
