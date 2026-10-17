"""FastAPI entry point: web UI, the jury API (/api/v1, X-API-Key) and keyless UI endpoints (/ui)."""

import hmac
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser, parse_options_header

from app.config import settings
from app.jobs import DONE, FAILED, PROCESSING, JobStore, Worker, remove_file
from app.norms import catalog_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
# first atom of a QuickTime/MP4 file; AVI and Matroska are recognised by their own magic bytes
QT_ATOMS = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
# uploads are parsed by hand (see _read_upload), so their body is declared for /docs here
UPLOAD_BODY = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file"],
    "properties": {"file": {"type": "string", "contentMediaType": "application/octet-stream"}},
}}}}}

store = JobStore(settings.storage_dir, keep_videos=settings.keep_videos)


def _process(job_id: str, video_path: Path, progress) -> dict:
    from app.pipeline.analyze import analyze_video  # heavy imports stay out of web startup

    return analyze_video(video_path, progress=progress)


worker = Worker(store, _process)


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
    # compare raw header bytes: compare_digest raises on non-ASCII str, and Starlette decodes headers as latin-1
    provided = (x_api_key or "").encode("latin-1", "replace")
    if not provided or not any(hmac.compare_digest(provided, k.encode()) for k in settings.api_keys):
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


class _UploadParser(MultiPartParser):
    """Starlette's parser, except that a 'file' part sent without filename= still arrives as a file
    (Starlette would make it a text field capped at 1 MB), and spooled parts are dropped on any error."""

    def on_headers_finished(self) -> None:
        disposition = self._current_part.content_disposition or b""
        _, options = parse_options_header(disposition)
        if options.get(b"name") == b"file" and b"filename" not in options:
            self._current_part.content_disposition = disposition + b'; filename=""'
        super().on_headers_finished()

    async def parse(self) -> FormData:
        try:
            return await super().parse()
        except BaseException:  # Starlette closes them only on its own parse errors, not on our 413
            for spooled in self._files_to_close_on_error:
                spooled.close()
            raise


def _too_large() -> HTTPException:
    return HTTPException(status_code=413, detail=f"file larger than {settings.max_upload_mb} MB")


async def _read_upload(request: Request) -> FormData:
    """Parse the multipart body here rather than in FastAPI, which spools all of it before any check runs.
    The key dependency has already run; an oversized body is refused by Content-Length or while it streams."""
    cap = (settings.max_upload_mb + 1) * 1024 * 1024  # the file limit plus room for the multipart framing
    declared = request.headers.get("content-length", "")
    if declared.isdecimal() and int(declared) > cap:
        raise _too_large()
    if parse_options_header(request.headers.get("content-type", ""))[0].lower() != b"multipart/form-data":
        raise HTTPException(status_code=422, detail="expected multipart/form-data with the video in field 'file'")

    async def capped():
        seen = 0
        async for chunk in request.stream():
            seen += len(chunk)
            if seen > cap:
                raise _too_large()
            yield chunk

    try:
        return await _UploadParser(request.headers, capped()).parse()
    except HTTPException:
        raise
    except MultiPartException as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except Exception as exc:  # malformed body or client gone, reported as FastAPI's own form parsing does
        log.warning("could not parse upload: %r", exc)
        raise HTTPException(status_code=400, detail="There was an error parsing the body") from exc


async def _upload(request: Request) -> dict:
    form = await _read_upload(request)
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(status_code=422, detail="multipart field 'file' with the video is required")
        return await _accept_upload(file)
    finally:
        await form.close()


def _looks_like_video(head: bytes) -> bool:
    return head[4:8] in QT_ATOMS or (head[:4] == b"RIFF" and head[8:12] == b"AVI ") or head[:4] == b"\x1a\x45\xdf\xa3"


async def _accept_upload(file: UploadFile) -> dict:
    head = await file.read(12)
    await file.seek(0)
    if not head:
        raise HTTPException(status_code=400, detail="empty file")
    if not file.filename and not _looks_like_video(head):  # no name to check, so the content has to show a video
        raise HTTPException(status_code=400, detail="field 'file' has no filename and does not look like a video")
    name = file.filename or "video.mov"
    suffix = Path(name).suffix.lower() or ".mov"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"unsupported file type {suffix}; expected .mov")
    job_id = store.new_job(name)["job_id"]
    dest = store.video_path(job_id, suffix)
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise _too_large()
                out.write(chunk)
    except HTTPException as exc:
        _drop_upload(job_id, dest, exc.detail)
        raise
    except Exception as exc:  # e.g. disk full: no job may stay 'queued' without its video
        log.exception("job %s: could not store the upload", job_id)
        _drop_upload(job_id, dest, "upload could not be stored")
        raise HTTPException(status_code=507, detail="could not store the upload") from exc
    store.update(job_id, size_bytes=size)
    worker.submit(job_id, dest)
    log.info("job %s queued: %s (%.1f MB)", job_id, name, size / 1e6)
    return {"job_id": job_id, "status": PROCESSING}


def _drop_upload(job_id: str, dest: Path, error: str) -> None:
    remove_file(dest)  # first, so that on a full disk the job record has room to be saved
    store.update(job_id, persist=True, status=FAILED, stage="failed", error=error)


def _job_or_404(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# ---- jury API -------------------------------------------------------------------------------

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)], openapi_extra=UPLOAD_BODY)
async def api_analyze(request: Request):
    return await _upload(request)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def api_job(job_id: str):
    return public_view(_job_or_404(job_id))


@app.get("/api/v1/health")
def health():
    alive = worker.alive
    body = {"status": "ok" if alive else "degraded", "queue": worker.queue.qsize(), "worker_alive": alive}
    return body if alive else JSONResponse(status_code=503, content=body)


@app.get("/api/v1/violation-types")
def violation_types():
    return catalog_for_api()


# ---- web UI (no auth, see FAQ 27) --------------------------------------------------------------

@app.post("/ui/analyze", openapi_extra=UPLOAD_BODY)
async def ui_analyze(request: Request):
    return await _upload(request)


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
