"""API contract tests (section 10 of the case). The video pipeline is replaced by a stub, so no GPU is needed."""

import asyncio
import base64
import dataclasses
import json
import os
import tempfile
import threading
import time
from tempfile import SpooledTemporaryFile

os.environ["API_KEYS"] = "test-key"
os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="dt26-test-")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette import formparsers  # noqa: E402

import app.main as main  # noqa: E402

KEY = {"X-API-Key": "test-key"}
JPEG = base64.b64encode(b"\xff\xd8\xff\xd9").decode()


def fake_process(job_id, video_path, progress):
    progress(0.5, "analyzing")
    return {
        "verdict": "violation",
        "violations": [{
            "type": "Сейф оставлен открытым",
            "timecode_sec": 6.4,
            "description": "Дверца сейфа открыта, кассир сидит за столом.",
            "evidence_frames": [{"timestamp_sec": 6.4, "image_base64": JPEG}],
            "norm": {"document": "НБРК № 231", "clause": "п. 20", "url": "https://adilet.zan.kz", "quote": "…",
                     "interpretation": None},
        }],
    }


@pytest.fixture(scope="module")
def client():
    main._warmup = lambda: None          # do not load models
    main.worker.process = fake_process
    with TestClient(main.app) as c:
        yield c


def upload(client, name="video.mov", data=b"\x00\x00\x00\x14ftypqt  ", headers=KEY):
    return client.post("/api/v1/analyze", headers=headers, files={"file": (name, data, "video/quicktime")})


def wait_done(client, job_id, headers=KEY, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        if r.json()["status"] != "processing":
            return r
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_key_is_required(client):
    assert upload(client, headers={}).status_code == 401
    assert upload(client, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/jobs/abc").status_code == 401


def test_full_flow_matches_schema(client):
    r = upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processing" and body["job_id"]
    done = wait_done(client, body["job_id"]).json()
    assert done["status"] == "done"
    assert done["verdict"] in ("violation", "no_violation")
    v = done["violations"][0]
    for field in ("type", "timecode_sec", "description", "evidence_frames"):
        assert field in v
    assert 1 <= len(v["evidence_frames"]) <= 3
    frame = v["evidence_frames"][0]
    assert isinstance(frame["timestamp_sec"], float)
    base64.b64decode(frame["image_base64"])


def test_ui_endpoints_work_without_key(client):
    r = client.post("/ui/analyze", files={"file": ("v.mov", b"\x00\x01", "video/quicktime")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    end = time.time() + 10
    while client.get(f"/ui/jobs/{job_id}").json()["status"] == "processing" and time.time() < end:
        time.sleep(0.05)
    assert client.get(f"/ui/jobs/{job_id}").json()["status"] == "done"


def test_rejects_non_video_and_empty(client):
    assert upload(client, name="notes.txt").status_code == 400
    assert upload(client, data=b"").status_code == 400


def test_unknown_job(client):
    assert client.get("/api/v1/jobs/doesnotexist", headers=KEY).status_code == 404


def test_violation_catalog_has_norms(client):
    types = client.get("/api/v1/violation-types").json()["types"]
    assert len(types) >= 10
    assert all(t["norms"] and t["norms"][0]["url"].startswith("https://adilet.zan.kz/") for t in types)


def test_health_reports_a_dead_worker(client, monkeypatch):
    assert client.get("/api/v1/health").json()["worker_alive"] is True
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    monkeypatch.setattr(main.worker, "thread", dead)
    r = client.get("/api/v1/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded" and r.json()["worker_alive"] is False


def test_non_ascii_key_gets_401_not_500(client):
    key = {"X-API-Key": "ключ".encode()}
    assert upload(client, headers=key).status_code == 401
    assert client.get("/api/v1/jobs/abc", headers=key).status_code == 401


def test_file_part_without_filename_is_taken_as_mov_if_it_is_a_video(client):
    video = b"\x00\x00\x00\x14ftypqt  " + bytes(2 * 1024 * 1024)  # over Starlette's 1 MB text-field limit
    r = client.post("/api/v1/analyze", headers=KEY, files={"file": (None, video)})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert wait_done(client, job_id).json()["status"] == "done"
    assert main.store.get(job_id)["filename"] == "video.mov"
    assert client.post("/api/v1/analyze", headers=KEY, files={"file": (None, b"just text")}).status_code == 400


def test_body_without_a_file_part_is_rejected(client):
    assert client.post("/api/v1/analyze", headers=KEY, files={"other": ("v.mov", b"\x00\x01")}).status_code == 422
    assert client.post("/api/v1/analyze", headers=KEY, json={"file": "v.mov"}).status_code == 422


def test_storage_error_during_upload_leaves_no_queued_job(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.store, "video_path", lambda job_id, suffix: tmp_path / "gone" / f"{job_id}{suffix}")
    before = set(main.store.jobs_dir.glob("*.json"))
    assert upload(client).status_code == 507
    (path,) = set(main.store.jobs_dir.glob("*.json")) - before
    record = json.loads(path.read_text(encoding="utf-8"))
    assert (record["status"], record["stage"]) == ("failed", "failed")
    assert main.store.queue_position(record["job_id"]) == -1


def post_raw(path, headers, body, chunk=64 * 1024):
    """Call the ASGI app directly; returns the status and how many body bytes the app pulled."""
    chunks = [body[i:i + chunk] for i in range(0, len(body), chunk)]
    pulled, status = 0, None

    async def receive():
        nonlocal pulled
        part = chunks.pop(0) if chunks else b""
        pulled += len(part)
        return {"type": "http.request", "body": part, "more_body": bool(chunks)}

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
             "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
             "client": ("testclient", 50000), "server": ("testserver", 80)}
    asyncio.run(main.app(scope, receive, send))
    return status, pulled


def test_key_and_size_are_checked_before_the_body_is_read(monkeypatch):
    class Spool(SpooledTemporaryFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spooled.append(self)

    spooled = []
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", Spool)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, max_upload_mb=1))
    body = (b'--dt26\r\nContent-Disposition: form-data; name="file"; filename="v.mov"\r\n\r\n'
            + bytes(3 * 1024 * 1024) + b"\r\n--dt26--\r\n")
    ctype = {"content-type": "multipart/form-data; boundary=dt26"}
    sized = {**ctype, "content-length": str(len(body))}
    assert post_raw("/api/v1/analyze", sized, body) == (401, 0)
    assert post_raw("/api/v1/analyze", {**sized, "x-api-key": "wrong"}, body) == (401, 0)
    assert post_raw("/api/v1/analyze", {**sized, "x-api-key": "test-key"}, body) == (413, 0)
    assert not spooled
    status, pulled = post_raw("/ui/analyze", ctype, body)  # no Content-Length: cut off while it streams in
    assert status == 413 and pulled < len(body)
    assert spooled and all(f.closed for f in spooled)  # the part spooled so far is not left in %TEMP%


def test_openapi_still_documents_the_upload(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/v1/analyze", "/ui/analyze"):
        assert "file" in paths[path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
