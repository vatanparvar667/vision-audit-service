"""Job store and the single GPU worker that processes uploaded videos in FIFO order."""

import json
import logging
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

log = logging.getLogger("jobs")

# Public statuses are fixed by the API contract: processing | done | failed.
PROCESSING, DONE, FAILED = "processing", "done", "failed"


def remove_file(path: Path) -> None:
    """Best-effort delete: on Windows another process (antivirus, indexer, player) may still hold the file open."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not delete %s: %s", path, exc)


class JobStore:
    """Thread-safe job registry. Finished results are persisted as JSON so a restart keeps them."""

    def __init__(self, storage_dir: Path, keep_videos: bool = False):
        self.jobs_dir = storage_dir / "jobs"
        self.videos_dir = storage_dir / "videos"
        self.keep_videos = keep_videos
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                job_id = job["job_id"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if job.get("status") == PROCESSING:  # interrupted by a restart: it will never run, so finish it
                job.update(status=FAILED, stage="failed", error="service restarted during processing",
                           finished_at=time.time())
                self._persist(job)
            self._jobs[job_id] = job
        if not self.keep_videos:  # nothing is queued yet: uploads of interrupted jobs or deletes that failed
            for video in self.videos_dir.iterdir():
                if video.stem in self._jobs:
                    remove_file(video)

    def _persist(self, job: dict) -> None:
        self._write(job["job_id"], json.dumps(job, ensure_ascii=False))

    def _write(self, job_id: str, data: str) -> None:
        # memory stays authoritative: a failed save only matters if the service restarts
        tmp = self.jobs_dir / f"{job_id}.json.tmp"
        try:
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(self.jobs_dir / f"{job_id}.json")
        except OSError as exc:
            log.error("could not save job %s: %s", job_id, exc)

    def new_job(self, filename: str) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": PROCESSING,
            "stage": "queued",
            "progress": 0.0,
            "filename": filename,
            "created_at": time.time(),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        return job

    def video_path(self, job_id: str, suffix: str) -> Path:
        return self.videos_dir / f"{job_id}{suffix}"

    def release_video(self, path: Path) -> None:
        """Drop a processed upload unless KEEP_VIDEOS is set."""
        if not self.keep_videos:
            remove_file(path)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update(self, job_id: str, persist: bool = False, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            # serialised before it is applied: a result that cannot be saved changes nothing
            data = json.dumps({**job, **fields}, ensure_ascii=False) if persist else None
            job.update(fields)
        if data is not None:
            self._write(job_id, data)

    def queue_position(self, job_id: str) -> int:
        with self._lock:
            waiting = sorted(
                (j["created_at"], j["job_id"]) for j in self._jobs.values()
                if j.get("status") == PROCESSING and j.get("stage") == "queued"
            )
        return next((i for i, (_, jid) in enumerate(waiting) if jid == job_id), -1)


class Worker:
    """Runs `process(job_id, video_path, progress_cb) -> result dict` for queued jobs, one at a time."""

    def __init__(self, store: JobStore, process: Callable):
        self.store = store
        self.process = process
        self.queue: queue.Queue[tuple[str, Path]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="video-worker", daemon=True)

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def start(self) -> None:
        self.thread.start()

    def submit(self, job_id: str, video_path: Path) -> None:
        self.queue.put((job_id, video_path))

    def _run(self) -> None:
        while True:
            job_id, video_path = self.queue.get()
            try:
                self._handle(job_id, video_path)
            except BaseException:  # this is the only worker: nothing a job does may stop it
                log.exception("job %s: worker error", job_id)

    def _handle(self, job_id: str, video_path: Path) -> None:
        started = time.time()

        def progress(fraction: float, stage: str = "analyzing") -> None:
            self.store.update(job_id, progress=round(float(fraction), 3), stage=stage)

        try:
            self.store.update(job_id, stage="analyzing", started_at=started)
            result = self.process(job_id, video_path, progress)
            self.store.update(
                job_id, persist=True, status=DONE, stage="done", progress=1.0,
                finished_at=time.time(), processing_sec=round(time.time() - started, 2), **result,
            )
            log.info("job %s done in %.1fs: %s", job_id, time.time() - started, result.get("verdict"))
        except BaseException as exc:  # SystemExit included: report failure through the API instead of hanging
            log.error("job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            self.store.update(
                job_id, persist=True, status=FAILED, stage="failed",
                error=f"{type(exc).__name__}: {exc}", finished_at=time.time(),
            )
        finally:
            self.store.release_video(video_path)
