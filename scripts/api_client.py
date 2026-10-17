"""Minimal client for the jury API: upload a video, poll the job, print the result without image data.

Usage: python scripts/api_client.py BASE_URL API_KEY video.mov [more.mov ...]
"""

import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def post_video(base: str, key: str, path: Path) -> dict:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: video/quicktime\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{base}/api/v1/analyze", data=body, method="POST", headers={
        "X-API-Key": key, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def get_job(base: str, key: str, job_id: str) -> dict:
    req = urllib.request.Request(f"{base}/api/v1/jobs/{job_id}", headers={"X-API-Key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    base, key = sys.argv[1].rstrip("/"), sys.argv[2]
    for name in sys.argv[3:]:
        t0 = time.time()
        job = post_video(base, key, Path(name))
        while job["status"] == "processing":
            time.sleep(2)
            job = get_job(base, key, job["job_id"])
        for v in job.get("violations", []):
            for f in v.get("evidence_frames", []):
                f["image_base64"] = f"<{len(f['image_base64'])} chars>"
        print(f"== {name} ({time.time() - t0:.1f}s)")
        print(json.dumps(job, ensure_ascii=False, indent=2))
