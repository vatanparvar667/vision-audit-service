"""Turn verified events into the API result: verdict, violations with descriptions, evidence frames, norms."""

import base64

import cv2

from app.norms import VIOLATIONS, primary_norm
from app.pipeline.rules import Event, Timeline, _dedup
from app.pipeline.video import VideoClip

MIN_SCORE = 0.5          # events below this are not reported
MAX_EVENTS = 6
COLORS = [(40, 40, 230), (0, 170, 255), (255, 120, 0)]


def _fmt(t: float) -> str:
    tenths = int(round(max(0.0, t) * 10))
    m, rest = divmod(tenths, 600)
    return f"{m}:{rest // 10:02d}.{rest % 10}"


def describe(e: Event) -> str:
    f = e.facts
    d = {
        "cash_unattended": lambda: (
            f"Около {_fmt(e.t)} кассир покидает рабочее место. После этого на столе видны: {', '.join(f['items'])}; "
            f"людей в помещении нет не менее {f['empty_for']:.0f} с. Ценности остаются без присмотра вне закрывающегося хранилища."),
        "pc_unlocked": lambda: (
            f"Около {_fmt(e.t)} кассир уходит, а экран компьютера продолжает показывать рабочее приложение: "
            f"признаков блокировки экрана в кадре не видно."),
        "food_drink": lambda: (
            f"Человек за кассовым столом подносит к лицу чашку или бутылку (эпизод около {max(1.0, f['duration']):.0f} с). "
            f"Приём пищи и напитков происходит на рабочем месте с наличными."),
        "personal_items": lambda: (
            f"На кассовом столе лежит сумка или верхняя одежда; предмет виден не менее {f['span']:.0f} с."),
        "door_open": lambda: (
            f"После входа или выхода около {_fmt(f['t_entry'])} дверь помещения остаётся открытой "
            f"не менее {f['open_for']:.0f} с."),
        "safe_open": lambda: (
            f"Дверца сейфа открыта около {f['open_for']:.0f} с, при этом кассир "
            + ("покидает помещение" if f.get("empty_room") else "сидит за столом или находится в стороне от сейфа")
            + f" не менее {f['away_for']:.0f} с."),
        "prev_client_cash": lambda: (
            f"Клиент уходит около {_fmt(f['t_exit'])}, а наличные остаются открыто лежать на столе. "
            f"Около {_fmt(f['t_next'])} входит следующий посетитель, наличные всё ещё на столе."),
        "count_without_client": lambda: (
            f"Клиент покидает помещение около {_fmt(f['t_exit'])}, а наличные ещё находятся на столе или в работе у кассира: "
            f"операция с деньгами продолжается без клиента."),
        "second_visitor": lambda: (
            f"В помещении одновременно находятся {f['people']} человека (около {f['duration']:.0f} с): "
            f"к рабочему месту подходит ещё один посетитель, пока обслуживание предыдущего не завершено."),
        "bag_no_check": lambda: (
            f"Посетитель приносит сумку к кассовому столу и уходит через {f['stay']:.0f} с; "
            f"проверки сумки, пломбы и документов в кадре не видно."),
        "outsider": lambda: (
            f"Через открытую дверь в помещение входит человек и находится рядом с рабочим местом "
            f"не менее {f['stay']:.0f} с."),
    }
    return d[e.type]()


def confidence_label(score: float) -> str:
    return "высокая" if score >= 0.75 else "средняя" if score >= 0.6 else "низкая"


def evidence_frames(clip: VideoClip, e: Event) -> list[dict]:
    out = []
    seen = set()
    for t in e.evidence_t[:3]:
        i = clip.nearest(t)
        if i in seen:
            continue
        seen.add(i)
        img = clip.bgr(i).copy()
        boxes = e.boxes.get(t) or []
        for k, (_, box) in enumerate(boxes):
            if box is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[k % len(COLORS)], 2)
        ts = clip.samples[i].t
        cv2.rectangle(img, (0, 0), (118, 28), (0, 0, 0), -1)
        cv2.putText(img, _fmt(ts), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append({"timestamp_sec": round(ts, 2), "image_base64": base64.b64encode(enc.tobytes()).decode("ascii")})
    return out


def build_report(clip: VideoClip, tl: Timeline, events: list[Event]) -> dict:
    reported = _dedup([e for e in events if e.score >= MIN_SCORE], window=3.0)
    reported = sorted(reported, key=lambda e: -e.score)[:MAX_EVENTS]
    reported.sort(key=lambda e: e.t)
    violations = []
    for e in reported:
        v = VIOLATIONS[e.type]
        violations.append({
            "type": v["name"],
            "timecode_sec": round(e.t, 2),
            "description": describe(e),
            "evidence_frames": evidence_frames(clip, e),
            "type_id": e.type,
            "confidence": confidence_label(e.score),
            "score": round(e.score, 3),
            "norm": primary_norm(e.type),
        })
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "video": {"duration_sec": round(clip.duration, 2), "width": clip.width, "height": clip.height},
    }
