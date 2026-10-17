"""Jury-style check of a deployed service against the case spec (ТЗ sections 4, 8, 10 and FAQ 13-18, 23, 25).

Does what the jury does: sends .mov videos one at a time to POST /api/v1/analyze, polls GET /api/v1/jobs/{job_id}
and counts a video as unprocessed when there is no result within 5 minutes of the upload start. Every answer is checked
against the schema from the ТЗ; with --labels verdicts and timecodes are scored. Full JSON answers and evidence frames
go to --out for a manual look. Standard library only, so it runs on any machine with Python 3.10+.

Usage:
  python scripts/jury_check.py BASE_URL API_KEY VIDEOS... [--labels runs/labels.json] [--out DIR] [--poll SEC]

VIDEOS are .mov files or folders with them. Labels: {"X-1": [22], "X-3": []} - violation timecodes per video, an empty
list for a clean one. Exit code is 0 only when every video got a result in time and every answer matches the ТЗ.
"""

import argparse
import base64
import binascii
import http.client
import ipaddress
import json
import re
import statistics
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

LIMIT_SEC = 300      # section 10: no result within 5 minutes -> the video counts as unprocessed
STATUSES = ("processing", "done", "failed")
VERDICTS = ("violation", "no_violation")
FRAME_GAP_SEC = 5.0  # an evidence frame further than this from its timecode is flagged for a look
NET_ERRORS = (OSError, http.client.HTTPException)


def call(method: str, url: str, headers: dict, body: bytes | None = None, timeout: float = 30) -> tuple[int, bytes]:
    """HTTP request; returns (status, body) for any HTTP status, raises NET_ERRORS when the server can't be reached."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def multipart(filename: str, data: bytes) -> tuple[bytes, str]:
    """Body and Content-Type of a form with one field `file`, as in the ТЗ example (curl -F "file=@video.mov")."""
    boundary = uuid.uuid4().hex
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: video/quicktime\r\n\r\n").encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def parse_json(raw: bytes) -> dict | None:
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def norm(name: str) -> str:
    """Video key without extension; dataset names mix Latin X and Cyrillic Х (X-1.mov, Х-10.mov)."""
    return Path(name).stem.replace("Х", "X").replace("х", "x")


def natural(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", norm(path.name))]


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def mov_duration(path: Path) -> float | None:
    """Movie duration from the QuickTime 'mvhd' atom, None if it can't be read."""
    try:
        with path.open("rb") as f:
            end, pos = path.stat().st_size, 0
            while pos + 8 <= end:
                f.seek(pos)
                size, kind = struct.unpack(">I4s", f.read(8))
                head = 8
                if size == 1:
                    size, head = struct.unpack(">Q", f.read(8))[0], 16
                elif size == 0:
                    size = end - pos
                if size < head:
                    return None
                if kind == b"moov":
                    moov = f.read(size - head)
                    i = moov.find(b"mvhd")
                    if i < 0:
                        return None
                    v1 = moov[i + 4] == 1  # version 1 has 64-bit times
                    scale, dur = struct.unpack_from(">IQ" if v1 else ">II", moov, i + (24 if v1 else 16))
                    return dur / scale if scale else None
                pos += size
    except (OSError, struct.error, IndexError):
        return None
    return None


def decode_jpeg(value) -> tuple[bytes | None, str | None]:
    """Decoded JPEG or an error; the ТЗ expects plain base64 of a JPEG, like "/9j/4AAQ..."."""
    if not isinstance(value, str) or not value:
        return None, "пустой или не строка"
    if value.startswith("data:"):
        return None, "с префиксом data:...;base64, а по ТЗ нужен чистый base64"
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None, "не декодируется как base64"
    if not raw.startswith(b"\xff\xd8\xff"):
        return None, "декодируется, но это не JPEG"
    return raw, None


def check_time(t: float, duration: float | None, where: str) -> list[str]:
    if t < 0 or (duration and t > duration + 0.5):
        span = f"0–{duration:.1f} с" if duration else "от 0"
        return [f"{where}={t} вне видео ({span})"]
    return []


def check_answer(data: dict, job_id, duration: float | None) -> tuple[list[str], list[str]]:
    """Errors break the ТЗ schema (automatic scoring may fail on them); warnings are worth a look."""
    errors, warnings = [], []
    if str(data.get("job_id")) != str(job_id):
        warnings.append(f"job_id в ответе {data.get('job_id')!r}, ожидался {job_id!r}")
    verdict, violations = data.get("verdict"), data.get("violations")
    if verdict not in VERDICTS:
        errors.append(f"verdict={verdict!r}, по ТЗ только violation / no_violation")
    if not isinstance(violations, list):
        errors.append("violations отсутствует или не список")
        return errors, warnings
    if verdict == "violation" and not violations:
        errors.append("verdict=violation, но violations пустой")
    if verdict == "no_violation" and violations:
        errors.append("verdict=no_violation, но violations не пустой (по ТЗ пустой, если нарушений нет)")
    for i, v in enumerate(violations, 1):
        where = f"violations[{i}]"
        if not isinstance(v, dict):
            errors.append(f"{where} не объект")
            continue
        for field in ("type", "description"):
            if not isinstance(v.get(field), str) or not v[field].strip():
                errors.append(f"{where}.{field} пустой или не строка")
        tc = v.get("timecode_sec")
        if is_number(tc):
            errors += check_time(tc, duration, f"{where}.timecode_sec")
        else:
            errors.append(f"{where}.timecode_sec={tc!r} не число")
            tc = None
        frames = v.get("evidence_frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            got = len(frames) if isinstance(frames, list) else repr(frames)
            errors.append(f"{where}.evidence_frames: по ТЗ 1–3 кадра, получено {got}")
            continue
        for j, frame in enumerate(frames, 1):
            fw = f"{where}.evidence_frames[{j}]"
            if not isinstance(frame, dict):
                errors.append(f"{fw} не объект")
                continue
            ts = frame.get("timestamp_sec")
            if not is_number(ts):
                errors.append(f"{fw}.timestamp_sec={ts!r} не число")
            else:
                errors += check_time(ts, duration, f"{fw}.timestamp_sec")
                if tc is not None and abs(ts - tc) > FRAME_GAP_SEC:
                    warnings.append(f"{fw}: кадр на {ts:.1f} с, а таймкод {tc:.1f} с")
            _, problem = decode_jpeg(frame.get("image_base64"))
            if problem:
                errors.append(f"{fw}.image_base64 {problem}")
    return errors, warnings


def address_warning(base: str) -> str | None:
    host = urllib.parse.urlsplit(base).hostname or ""
    try:
        private = host == "localhost" or ipaddress.ip_address(host).is_private
    except ValueError:  # a domain name
        return None
    return f"{host} - внутренний адрес, жюри проверяет удалённо (FAQ 13, 23) и до него не достучится" if private else None


def check_service(base: str, key: str) -> list[str]:
    """Web UI (sections 4, 8) and key rejection (section 10). Raises NET_ERRORS when the service is down."""
    problems = []
    status, body = call("GET", f"{base}/", {})
    if status != 200 or b"<html" not in body.lower():
        problems.append(f"веб-интерфейс: GET / вернул {status} без HTML-страницы")
    for what, headers in (("без ключа", {}), ("с неверным ключом", {"X-API-Key": f"wrong-{uuid.uuid4().hex[:8]}"})):
        status, _ = call("GET", f"{base}/api/v1/jobs/{uuid.uuid4().hex[:12]}", headers)
        if status not in (401, 403):
            problems.append(f"GET /api/v1/jobs {what} вернул {status}, а должен отклоняться (401/403)")
        body, ctype = multipart("probe.mov", b"not a video")
        status, _ = call("POST", f"{base}/api/v1/analyze", {**headers, "Content-Type": ctype}, body)
        if status not in (401, 403):
            problems.append(f"POST /api/v1/analyze {what} вернул {status}, а должен отклоняться (401/403)")
    return problems


def run_video(base: str, key: str, path: Path, poll: float) -> dict:
    """Upload and wait the way the jury does; the full answer is kept in rec['answer']."""
    rec = {"video": path.name, "outcome": "unprocessed", "verdict": None, "violations": [], "errors": [],
           "warnings": [], "duration_sec": mov_duration(path), "elapsed_sec": None, "answer": None}
    if path.suffix.lower() != ".mov":
        rec["warnings"].append("не .mov, а по ТЗ жюри шлёт только .mov")
    body, ctype = multipart(path.name, path.read_bytes())
    t0 = time.monotonic()
    try:
        status, raw = call("POST", f"{base}/api/v1/analyze", {"X-API-Key": key, "Content-Type": ctype}, body, LIMIT_SEC)
    except NET_ERRORS as e:
        rec["errors"].append(f"загрузка не прошла: {e}")
        return rec
    answer = parse_json(raw) if 200 <= status < 300 else None
    job_id = answer.get("job_id") if answer else None
    if not job_id:
        rec["errors"].append(f"POST /api/v1/analyze вернул {status} без job_id: {raw[:200]!r}")
        return rec
    if status != 200:
        rec["warnings"].append(f"POST /api/v1/analyze вернул {status}; ТЗ код не задаёт, надёжнее 200")
    rec["job_id"] = job_id

    data, poll_errors = None, []
    while time.monotonic() - t0 < LIMIT_SEC:
        try:
            status, raw = call("GET", f"{base}/api/v1/jobs/{job_id}", {"X-API-Key": key})
            data = parse_json(raw) if status == 200 else None
            if data is None:
                poll_errors.append(f"HTTP {status}")
        except NET_ERRORS as e:
            data = None
            poll_errors.append(str(e))
        if data is not None:
            if data.get("status") not in STATUSES:
                rec["errors"].append(f"status={data.get('status')!r}, по ТЗ только processing / done / failed")
                return rec
            if data["status"] != "processing":
                break
        time.sleep(poll)
    rec["elapsed_sec"] = round(time.monotonic() - t0, 1)
    if poll_errors:
        rec["warnings"].append(f"{len(poll_errors)} неудачных запросов статуса, последний: {poll_errors[-1]}")

    if data is None or data["status"] == "processing" or rec["elapsed_sec"] > LIMIT_SEC:
        rec["errors"].append(f"нет результата за {LIMIT_SEC // 60} минут, засчитывается как необработанное")
        return rec
    rec["answer"] = data
    if data["status"] == "failed":
        rec["outcome"] = "failed"
        rec["errors"].append(f"status=failed: {data.get('error')}")
        return rec
    rec["outcome"] = "done"
    rec["verdict"] = data.get("verdict")
    errors, warnings = check_answer(data, job_id, rec["duration_sec"])
    rec["errors"] += errors
    rec["warnings"] += warnings
    if isinstance(data.get("violations"), list):
        rec["violations"] = [(v.get("timecode_sec"), v.get("type")) for v in data["violations"] if isinstance(v, dict)]
    return rec


def save(rec: dict, out: Path) -> None:
    """Full answer as JSON and decodable evidence frames as .jpg next to it."""
    if rec["answer"] is None:
        return
    stem = norm(rec["video"])
    (out / f"{stem}.json").write_text(json.dumps(rec["answer"], ensure_ascii=False, indent=2), encoding="utf-8")
    violations = rec["answer"].get("violations")
    for i, v in enumerate(violations if isinstance(violations, list) else [], 1):
        frames = v.get("evidence_frames") if isinstance(v, dict) else None
        for j, frame in enumerate(frames if isinstance(frames, list) else [], 1):
            raw, _ = decode_jpeg(frame.get("image_base64") if isinstance(frame, dict) else None)
            if raw:
                (out / f"{stem}_v{i}_f{j}.jpg").write_bytes(raw)


def print_record(n: int, total: int, rec: dict) -> None:
    took = f"{rec['elapsed_sec']:.1f} с" if rec["elapsed_sec"] is not None else "—"
    head = {"done": rec["verdict"], "failed": "FAILED", "unprocessed": "НЕТ РЕЗУЛЬТАТА"}[rec["outcome"]]
    print(f"[{n}/{total}] {rec['video']}  {took}  {head}")
    for tc, kind in rec["violations"]:
        print(f"      {tc:.1f} с  {kind}" if is_number(tc) else f"      {tc!r}  {kind}")
    for e in rec["errors"]:
        print(f"    ошибка: {e}")
    for w in rec["warnings"]:
        print(f"    внимание: {w}")


def score(records: list[dict], labels: dict) -> dict | None:
    """Verdict per video (FAQ 15) with `violation` as the positive class; no result counts as a wrong answer."""
    tp = fp = fn = tn = 0
    errs, missed, false_alarms = [], [], []
    for r in records:
        gt = labels.get(norm(r["video"]))
        if gt is None:
            continue
        truth, got = bool(gt), r["outcome"] == "done" and r["verdict"] in VERDICTS
        pred = r["verdict"] == "violation" if got else not truth
        name = r["video"] if got else f"{r['video']} (нет результата)"
        if truth and pred:
            tp += 1
            times = [tc for tc, _ in r["violations"] if is_number(tc)]
            errs += [min(abs(tc - t) for tc in times) for t in gt if times]
        elif truth:
            fn += 1
            missed.append(name)
        elif pred:
            fp += 1
            false_alarms.append(name)
        else:
            tn += 1
    n = tp + fp + fn + tn
    if not n:
        print("\nРазметка: ни одно видео не нашлось в файле меток")
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\nРазметка ({n} видео): верный вердикт {tp + tn}/{n} ({(tp + tn) / n:.0%})")
    print(f"Нарушение как положительный класс: precision {precision:.2f}, recall {recall:.2f}, F1 {f1:.2f}")
    print(f"Пропуски: {', '.join(missed) or 'нет'}")
    print(f"Ложные срабатывания: {', '.join(false_alarms) or 'нет'}")
    if errs:
        print(f"Ошибка таймкода по найденным: медиана {statistics.median(errs):.1f} с, макс {max(errs):.1f} с")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / n, "precision": precision,
            "recall": recall, "f1": f1, "missed": missed, "false_alarms": false_alarms,
            "timecode_error_median": statistics.median(errs) if errs else None}


def collect(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted((q for q in p.iterdir() if q.suffix.lower() == ".mov"), key=natural)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"нет такого файла или папки: {p}")
    return files


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Проверка сервиса так, как её проводит жюри (ТЗ, раздел 10)")
    ap.add_argument("base_url", help="публичный адрес сервиса, например https://team.example.kz")
    ap.add_argument("api_key", help="ключ, который передаётся жюри")
    ap.add_argument("videos", nargs="+", type=Path, help=".mov файлы или папки с ними")
    ap.add_argument("--labels", type=Path, help='JSON {"X-1": [22], "X-3": []}: таймкоды нарушений, [] если чисто')
    ap.add_argument("--out", type=Path, default=Path("runs/jury_check") / time.strftime("%Y%m%d-%H%M%S"),
                    help="куда сохранить ответы и кадры")
    ap.add_argument("--poll", type=float, default=1.0, help="секунд между запросами статуса")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    videos = collect(args.videos)
    if not videos:
        print("нет .mov файлов")
        return 2
    labels = None
    if args.labels:
        labels = {norm(k): v for k, v in json.loads(args.labels.read_text(encoding="utf-8")).items()}
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Сервис: {base}  ·  видео: {len(videos)}  ·  лимит {LIMIT_SEC // 60} мин на видео")
    if warning := address_warning(base):
        print(f"внимание: {warning}")
    try:
        problems = check_service(base, args.api_key)
    except NET_ERRORS as e:
        print(f"ошибка: сервис недоступен: {e}")
        return 1
    for p in problems:
        print(f"ошибка: {p}")
    if not problems:
        print("Интерфейс открывается, запросы без верного ключа отклоняются")
    print()

    records = []
    for n, path in enumerate(videos, 1):
        rec = run_video(base, args.api_key, path, args.poll)
        save(rec, args.out)
        print_record(n, len(videos), rec)
        records.append(rec)

    done = [r for r in records if r["outcome"] == "done"]
    lost = [r["video"] for r in records if r["outcome"] != "done"]
    valid = [r for r in done if not r["errors"]]
    print("\n=== Итог ===")
    print(f"Результат за {LIMIT_SEC // 60} мин: {len(done)}/{len(records)}"
          + (f"  (нет результата: {', '.join(lost)})" if lost else ""))
    print(f"Ответ по схеме ТЗ: {len(valid)}/{len(done)}")
    times = [r["elapsed_sec"] for r in done]
    if times:
        print(f"Время на видео: медиана {statistics.median(times):.1f} с, макс {max(times):.1f} с")
    metrics = score(records, labels) if labels else None
    summary = {"base_url": base, "service_problems": problems, "metrics": metrics,
               "videos": [{k: v for k, v in r.items() if k != "answer"} for r in records]}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ответы и кадры: {args.out.resolve()}")
    return 0 if not problems and not lost and len(valid) == len(done) else 1


if __name__ == "__main__":
    sys.exit(main())
