"""API contract tests (section 10 of the case). The video pipeline is replaced by a stub, so no GPU is needed."""

import base64
import os
import tempfile
import time

os.environ["API_KEYS"] = "test-key"
os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="dt26-test-")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

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
