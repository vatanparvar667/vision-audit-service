"""JobStore and Worker robustness: the single worker survives anything a job does, and a restart cleans up."""

import json
import time
from pathlib import Path

from app.jobs import DONE, FAILED, PROCESSING, JobStore, Worker


def ok(job_id, video_path, progress):
    return {"verdict": "no_violation", "violations": []}


def submit(store, worker, name="v.mov"):
    job_id = store.new_job(name)["job_id"]
    path = store.video_path(job_id, ".mov")
    path.write_bytes(b"\x00\x00\x00\x14ftypqt  ")
    worker.submit(job_id, path)
    return job_id, path


def wait_finished(store, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        job = store.get(job_id)
        if job["status"] != PROCESSING:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def on_disk(store, job_id):
    return json.loads((store.jobs_dir / f"{job_id}.json").read_text(encoding="utf-8"))


def test_worker_survives_base_exception_from_process(tmp_path):
    def process(job_id, video_path, progress):
        if not calls:
            calls.append(job_id)
            raise SystemExit(3)
        return ok(job_id, video_path, progress)

    calls = []
    store = JobStore(tmp_path)
    worker = Worker(store, process)
    worker.start()
    first, first_video = submit(store, worker)
    second, _ = submit(store, worker)
    assert wait_finished(store, first)["status"] == FAILED
    assert wait_finished(store, second)["status"] == DONE
    assert worker.alive and not first_video.exists()


def test_worker_survives_undeletable_video_and_restart_sweeps_it(tmp_path, monkeypatch):
    real_unlink = Path.unlink

    def held_open(self, missing_ok=False):  # what Windows does while another process has the file open
        if self.suffix == ".mov":
            raise PermissionError(32, "The process cannot access the file because it is being used")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held_open)
    store = JobStore(tmp_path)
    worker = Worker(store, ok)
    worker.start()
    first, video = submit(store, worker)
    second, _ = submit(store, worker)
    assert wait_finished(store, first)["status"] == DONE
    assert wait_finished(store, second)["status"] == DONE
    assert worker.alive and video.exists()
    monkeypatch.undo()
    JobStore(tmp_path)  # the next start removes what could not be deleted
    assert not video.exists()


def test_failed_save_keeps_the_computed_result(tmp_path, monkeypatch):
    def disk_full(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    store = JobStore(tmp_path)
    worker = Worker(store, ok)
    worker.start()
    monkeypatch.setattr(Path, "write_text", disk_full)
    job_id, _ = submit(store, worker)
    job = wait_finished(store, job_id)
    assert job["status"] == DONE and job["verdict"] == "no_violation"
    assert worker.alive


def test_unsaveable_result_fails_the_job_cleanly(tmp_path):
    store = JobStore(tmp_path)
    worker = Worker(store, lambda *_: {"verdict": object(), "violations": []})
    worker.start()
    job_id, _ = submit(store, worker)
    job = wait_finished(store, job_id)
    assert job["status"] == FAILED and job["error"].startswith("TypeError") and "verdict" not in job
    worker.process = ok
    assert wait_finished(store, submit(store, worker)[0])["status"] == DONE
    assert on_disk(store, job_id)["status"] == FAILED  # read only now: the worker has moved on from it


def test_restart_finishes_interrupted_jobs(tmp_path):
    store = JobStore(tmp_path)
    running = store.new_job("a.mov")["job_id"]
    waiting = store.new_job("b.mov")["job_id"]
    store.update(running, stage="analyzing")
    for job_id in (running, waiting):
        store.video_path(job_id, ".mov").write_bytes(b"video")

    store = JobStore(tmp_path)  # the service restarted while both were in flight
    for job_id in (running, waiting):
        assert (store.get(job_id)["status"], store.get(job_id)["stage"]) == (FAILED, "failed")
        assert on_disk(store, job_id)["status"] == FAILED
        assert not store.video_path(job_id, ".mov").exists()
    current = store.new_job("c.mov")["job_id"]
    store.update(current, stage="analyzing")
    assert store.queue_position(store.new_job("d.mov")["job_id"]) == 0


def test_restart_keeps_videos_when_asked(tmp_path):
    store = JobStore(tmp_path, keep_videos=True)
    video = store.video_path(store.new_job("a.mov")["job_id"], ".mov")
    video.write_bytes(b"video")
    JobStore(tmp_path, keep_videos=True)
    assert video.exists()


def test_unreadable_records_do_not_stop_startup(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "a.json").write_bytes(b"\xff\xfe not utf-8")
    (jobs / "b.json").write_text("[1, 2]")
    (jobs / "c.json").write_text('{"status": "done"}')
    store = JobStore(tmp_path)
    assert store.get("a") is None and store.queue_position("a") == -1
