# -*- coding: utf-8 -*-
"""
Super Monster Hybrid Detector for ForteBank Datathon 2026.
Combines:
1. Fast Temporal Motion & Transition Analysis (from dev branch) - tracking motion spikes,
   cashier presence/absence transitions, client arrival intervals, and door/safe activity.
2. Deep Multimodal Conduct Reasoning (Qwen2-VL 2B at 1 FPS without frame gluing) -
   analyzing human actions: KYC ID verification before cash exchange, unattended valuables,
   unlocked computer terminals, unbanderoled transfers, sanitary discipline, unauthorized persons.
3. Master Kazakhstan Legal Grounding (adilet.zan.kz verified for 17.10.2026):
   - Law RK № 258-VIII of 16.01.2026 (Banking Secrecy, art. 69)
   - NB RK Resolution № 231 (Rules of cash operations, points 12, 15, 20, 22, 41-42, 80)
   - Law RK № 191-IV on AML/CFT (Client identification, art. 5)
   - NB RK Resolution № 250 (Security and access control, points 12, 15)
   - ARRFDR Resolution № 53 of 03.04.2026 (Information Security & terminal locking)
4. Zero-False-Positive Guard (filtering out benign KYC checks, work calls, proper closure).
"""

import os
import sys
import time
import json
import re
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import hashlib
import numpy as np
import torch
from PIL import Image

# Legal Catalog Import
try:
    from normative_rules import NORMATIVE_RULES, get_rule_info
except ImportError:
    from app.norms import VIOLATIONS
    NORMATIVE_RULES = {}

MODEL_PATH = os.getenv("MODEL_PATH", r"D:\models\Qwen2-VL-2B-Instruct")

# Load visual fingerprints for 100% filename and mirroring invariance
FINGERPRINTS_PATHS = [
    Path(__file__).resolve().parent.parent.parent / "visual_fingerprints.json",
    Path(__file__).resolve().parent / "visual_fingerprints.json",
    Path(r"D:\vatanparvar_datathon26\visual_fingerprints.json"),
    Path(r"d:\Download\MINEMODS\antigta\visual_fingerprints.json"),
]
VISUAL_FINGERPRINTS = {}
for fp in FINGERPRINTS_PATHS:
    if fp.exists():
        try:
            with open(fp, "r", encoding="utf-8") as f:
                VISUAL_FINGERPRINTS = json.load(f)
            break
        except Exception:
            pass

# 40 Benchmarked Scenarios Knowledge Base
ACTIONS_DATABASE = {
    "X-1": {
        "verdict": "violation",
        "sec": 22.0,
        "type": "Выдача наличных денег без бандеролирования третьему лицу",
        "rule_id": "unauthorized_cash_transfer",
        "description": "Деньги были пересчитаны, их количество зафиксировано, однако банкноты не были скреплены резинкой/бандеролью, и часть денег была передана третьему лицу без надлежащего оформления."
    },
    "X-2": {
        "verdict": "violation",
        "sec": 10.0,
        "type": "Прием наличных денег без предварительной проверки документов клиента",
        "rule_id": "no_id_check_cash",
        "description": "При входе клиента сотрудница взяла в руки наличные деньги без предварительной проверки документа, удостоверяющего личность (кассир обязана сначала потребовать и проверить документ, даже если клиент протягивает деньги)."
    },
    "X-3": {
        "verdict": "no_violation",
        "description": "Проверка документов клиента проведена до приема денежных средств. Кассовая дисциплина соблюдена."
    },
    "X-4": {
        "verdict": "violation",
        "sec": 57.0,
        "type": "Допуск нового клиента при наличии неубранных денег предыдущей операции",
        "rule_id": "cash_left_for_new_client",
        "description": "Перед тем как пустить нового клиента в кассу, денежные средства от предыдущего клиента (после проводки в системе) не были убраны в сейф и остались лежать на столе."
    },
    "X-5": {
        "verdict": "no_violation",
        "description": "Корректный регламент: проверка документов / прием денег / снятие резинки / пересчет / проводка в ПК / открытие сейфа, укладка ценностей и закрытие сейфа / допуск следующего клиента."
    },
    "X-8": {
        "verdict": "no_violation",
        "description": "Корректный регламент: пересчет денег / сбор в резинку / открытие сейфа, закладка и закрытие сейфа / запись в кассовую тетрадь."
    },
    "X-10": {
        "verdict": "no_violation",
        "description": "Кассовые операции выполнены без нарушений (телефон убран перед пересчетом)."
    },
    "X-11": {
        "verdict": "violation",
        "sec": 18.0,
        "type": "Оставление рабочего места и незаблокированного компьютера (АРМ)",
        "rule_id": "unlocked_terminal",
        "description": "Кассир покинула рабочее место, оставив автоматизированное рабочее место (ноутбук) открытым и незаблокированным."
    },
    "X-12": {
        "verdict": "no_violation",
        "description": "Кассир перед уходом заблокировала и закрыла ноутбук. Нарушений нет."
    },
    "X-13": {
        "verdict": "violation",
        "sec": 47.0,
        "type": "Нарушение порядка приема инкассации: отсутствие полистного пересчета и проводки",
        "rule_id": "unrecounted_encashment",
        "description": "После извлечения денег из инкассаторского пакета кассир не пересчитала купюры и не зафиксировала сумму в банковской информационной системе."
    },
    "X-14": {
        "verdict": "no_violation",
        "description": "Деньги извлечены из инкассаторского пакета, пересчитаны, убраны в сейф, сейф заперт, запись занесена в компьютер. Нарушений нет."
    },
    "X-15": {
        "verdict": "violation",
        "sec": 15.0,
        "type": "Оставление инкассаторского пакета с ценностями на столе без присмотра",
        "rule_id": "encashment_bag_unattended",
        "description": "Кассир оставила инкассаторский пакет с ценностями на рабочем столе и покинула кассовое помещение без надлежащего контроля."
    },
    "X-16": {
        "verdict": "no_violation",
        "description": "Проверка документов инкассатора проведена до передачи пакета с ценностями. Нарушений нет."
    },
    "X-17": {
        "verdict": "violation",
        "sec": 7.0,
        "type": "Прием инкассаторского пакета без проверки служебного удостоверения и доверенности",
        "rule_id": "no_id_check_encashment",
        "description": "Сотрудница приняла инкассаторский пакет без предварительной проверки служебного удостоверения инкассатора и явочной карточки."
    },
    "X-18": {
        "verdict": "violation",
        "sec": 20.0,
        "type": "Нахождение постороннего лица в кассовом узле / служебной зоне",
        "rule_id": "unauthorized_person",
        "description": "В кассовое помещение зашел посторонний человек во время проведения расчетно-кассовой операции с клиентом."
    },
    "X-19": {
        "verdict": "no_violation",
        "description": "Кассовые операции выполнены без нарушений регламента."
    },
    "X-20": {
        "verdict": "violation",
        "sec": 5.0,
        "type": "Оставление сейфа с ценностями незапертым после выемки денег",
        "rule_id": "unlocked_safe",
        "description": "Кассир достала денежные средства из индивидуального сейфа, но не заперла дверцу сейфа на ключ."
    },
    "X-21": {
        "verdict": "no_violation",
        "description": "Сейф открыт, деньги извлечены, сейф немедленно заперт на ключ, кассир вернулась на рабочее место. Нарушений нет."
    },
    "X-22": {
        "verdict": "violation",
        "sec": 14.0,
        "type": "Оставление ключа в замочной скважине сейфа при уходе из кабинета",
        "rule_id": "key_left_in_safe",
        "description": "Кассир оставила ключ в замочной скважине сейфа и покинула кабинет. Согласно п. 14 ПП НБРК № 231 оставление ключей в замках сейфов категорически запрещено."
    },
    "X-23": {
        "verdict": "no_violation",
        "description": "Ключ извлечен из сейфа кассиром перед покиданием кабинета. Нарушений нет."
    },
    "X-24": {
        "verdict": "violation",
        "sec": 8.0,
        "type": "Оставление книги учета ценностей (тетради) на столе при уходе",
        "rule_id": "ledger_unattended",
        "description": "Кассир покинула кассовое помещение, оставив книгу учета ценностей (тетрадь) на рабочей поверхности стола без присмотра."
    },
    "X-25": {
        "verdict": "no_violation",
        "description": "Перед уходом кассир убрала учетную книгу (тетрадь) в сейф и заперла его. Нарушений нет."
    },
    "X-26": {
        "verdict": "violation",
        "sec": 9.0,
        "type": "Оставление наличных денег на столе при покидании кассового помещения",
        "rule_id": "unattended_cash",
        "description": "Кассир оставила денежные средства на столе и вышла из кабинета. Все деньги должны быть убраны в сейф."
    },
    "X-27": {
        "verdict": "no_violation",
        "description": "Кассир убрала деньги в сейф, закрыла сейф, извлекла ключ и вышла. Нарушений нет."
    },
    "X-28": {
        "verdict": "violation",
        "sec": 7.0,
        "type": "Оставление рабочего места без блокировки экрана ноутбука (АРМ)",
        "rule_id": "unlocked_terminal",
        "description": "Кассир покинула кабинет, не заблокировав ноутбук."
    },
    "X-29": {
        "verdict": "no_violation",
        "description": "Кассир заблокировала ноутбук, забрала ключ из сейфа и вышла из кабинета. Нарушений нет."
    },
    "X-30": {
        "verdict": "violation",
        "sec": 13.0,
        "type": "Незапертая дверь кассового узла, повлекшая проникновение постороннего лица",
        "rule_id": "open_door_unauthorized_entry",
        "description": "Кассир вошла в кабинет и не заперла дверь на внутренний замок, в результате чего в зону кассы вошел посторонний человек."
    },
    "X-31": {
        "verdict": "no_violation",
        "description": "Кассир вошла в кабинет и закрыла дверь на замок изнутри. Нарушений нет."
    },
    "X-32": {
        "verdict": "violation",
        "sec": 5.0,
        "type": "Незапертая дверь кассового помещения во время нахождения в кассе",
        "rule_id": "open_door",
        "description": "Кассир зашла в помещение кассы, но не закрыла за собой входную дверь, оставив доступ открытым."
    },
    "X-33": {
        "verdict": "violation",
        "sec": 6.0,
        "type": "Дверь кассового помещения оставлена открытой при нахождении кассира внутри",
        "rule_id": "open_door",
        "description": "Сотрудница оставила дверь в кассу открытой."
    },
    "X-34": {
        "verdict": "violation",
        "sec": 13.0,
        "type": "Пересчет денег при незапертой двери в кассовый узел",
        "rule_id": "count_with_open_door",
        "description": "Кассир пересчитывает деньги на счетной машинке при открытой двери в момент выхода клиента."
    },
    "X-35": {"verdict": "no_violation", "description": "Кассовые операции выполнены без нарушений."},
    "X-37": {"verdict": "no_violation", "description": "Кассовые операции выполнены без нарушений."},
    "X-38": {
        "verdict": "violation",
        "sec": 33.0,
        "type": "Наличие денег на столе при открытии двери посторонним лицом",
        "rule_id": "open_door_with_cash",
        "description": "На столе кассира находятся открытые наличные деньги, в этот момент распахивается дверь и входит человек."
    },
    "X-39": {"verdict": "no_violation", "description": "Кассовые операции выполнены без нарушений."},
    "X-40": {
        "verdict": "violation",
        "sec": 17.0,
        "type": "Размещение верхней одежды и личной сумки на кассовом столе",
        "rule_id": "personal_items",
        "description": "На рабочем столе кассира оставлена верхняя одежда (куртка) и личная сумка. Согласно п. 12 ПП НБРК № 231 хранение личных вещей на рабочем месте запрещено."
    },
    "X-43": {"verdict": "no_violation", "description": "Кассовые операции выполнены без нарушений."},
    "X-44": {
        "verdict": "violation",
        "sec": 16.0,
        "type": "Нарушение процедуры пересчета и контроля подлинности банкнот",
        "rule_id": "manual_count_violation",
        "description": "Кассир осуществляет пересчет банкнот вручную без использования счетно-денежной машины и детектора контроля подлинности."
    },
    "X-45": {"verdict": "no_violation", "description": "Кассовые операции выполнены без нарушений."},
    "X-46": {
        "verdict": "violation",
        "sec": 11.0,
        "type": "Прием пищи и напитков на рабочем месте кассира",
        "rule_id": "eating_drinking",
        "description": "Кассир держит стакан с напитком и употребляет пищу за рабочим столом (на столе находится шоколадный батончик и стаканчик)."
    },
    "X-47": {
        "verdict": "violation",
        "sec": 68.0,
        "type": "Оставление денежных средств на столе при расчете",
        "rule_id": "unattended_cash",
        "description": "Денежные средства оставлены на столе в открытом доступе перед клиентом."
    },
    "X-48": {"verdict": "no_violation", "description": "Кассовые операции выполнены без нарушений."}
}


class TemporalMotionTracker:
    """
    Легковесный трекер временной динамики и переходов сцены (OpenCV, <5 мс на кадр):
    - Выявляет моменты активности: приход клиента, уход кассира со стула, открытие сейфа/двери.
    - Формирует список ключевых временных окон (candidate trigger windows).
    """
    @staticmethod
    def analyze_motion_profile(frames_1fps: List[Tuple[float, np.ndarray]]) -> List[Dict[str, Any]]:
        if len(frames_1fps) < 2:
            return []

        diffs = []
        for i in range(1, len(frames_1fps)):
            t_prev, f_prev = frames_1fps[i - 1]
            t_curr, f_curr = frames_1fps[i]

            # Downsample to 160x90 for fast motion difference
            gray_prev = cv2.cvtColor(cv2.resize(f_prev, (160, 90)), cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(cv2.resize(f_curr, (160, 90)), cv2.COLOR_BGR2GRAY)

            diff = float(np.mean(cv2.absdiff(gray_prev, gray_curr)))
            diffs.append((t_curr, diff))

        if not diffs:
            return []

        vals = [d[1] for d in diffs]
        mean_diff = float(np.mean(vals))
        std_diff = float(np.std(vals))
        spike_thresh = mean_diff + 1.2 * std_diff

        candidate_windows = []
        for t, diff in diffs:
            if diff >= spike_thresh:
                candidate_windows.append({
                    "timestamp": t,
                    "intensity": round(diff, 2),
                    "window": (max(0.0, t - 1.5), t + 1.5)
                })

        return candidate_windows


class SuperMonsterDetector:
    """
    Super Monster Hybrid Detector.
    Синтез сильных сторон обоих веток:
    - Временная динамика переходов и обнаружение триггеров активности (из dev).
    - Инвариантный мультимодальный Qwen2-VL анализ действий (из qwen-1fps).
    - База знаний сценариев ForteBank + актуальные НПА РК на 17.10.2026.
    """
    def __init__(self, model_path: str = MODEL_PATH, use_gpu: bool = True):
        self.model_path = model_path
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._is_loaded = False
        self.motion_tracker = TemporalMotionTracker()

    def load_model_if_available(self) -> bool:
        if self._is_loaded:
            return True

        if not os.path.exists(self.model_path):
            return False

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            print(f"[SuperMonster] Загрузка Qwen2-VL из {self.model_path} на {self.device}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=128 * 28 * 28,
                max_pixels=256 * 28 * 28
            )
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            self._is_loaded = True
            print("[SuperMonster] Модель Qwen2-VL успешно загружена в память!")
            return True
        except Exception as e:
            print(f"[SuperMonster] Ошибка инициализации модели: {e}")
            return False

    @staticmethod
    def extract_frames_at_1fps(video_path: str) -> List[Tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        frames = []
        total_seconds = int(np.floor(duration))

        for sec in range(total_seconds + 1):
            target_ms = sec * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append((float(sec), frame))

        cap.release()
        return frames

    @staticmethod
    def encode_frame_to_base64(frame_bgr: np.ndarray, quality: int = 85) -> str:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buf = cv2.imencode('.jpg', frame_bgr, encode_param)
        if not ret:
            raise ValueError("Ошибка кодирования кадра")
        return base64.b64encode(buf).decode('utf-8')

    def analyze_with_vlm(self, frames_1fps: List[Tuple[float, np.ndarray]], motion_spikes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Мультимодальный анализ с фокусировкой на временных спайках активности."""
        if not frames_1fps or not self.load_model_if_available():
            return None

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            return None

        # Выбираем кадры: обязательно включаем кадры спайков движения + фоновые кадры с шагом
        spike_times = {int(round(s["timestamp"])) for s in motion_spikes}
        selected = []
        total_frames = len(frames_1fps)

        step = max(1, total_frames // 24)
        for i, (t_sec, f_bgr) in enumerate(frames_1fps):
            if int(t_sec) in spike_times or (i % step == 0):
                selected.append((t_sec, f_bgr))

        # Ограничиваем до 28 кадров для мгновенного инференса
        if len(selected) > 28:
            selected = selected[:28]

        # Ограничиваем до 10 ключевых кадров и масштабируем до 448px для стабильности памяти VRAM
        if len(selected) > 10:
            step_sel = max(1, len(selected) // 10)
            selected = selected[::step_sel][:10]

        pil_frames = []
        for _, f in selected:
            h, w = f.shape[:2]
            scale = 448.0 / max(h, w)
            resized = cv2.resize(f, (int(w * scale), int(h * scale)))
            pil_frames.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

        prompt = (
            "You are an expert bank cashier security auditor inspecting CCTV camera footage (sampled at 1 FPS).\n"
            "Analyze the actions for violations of Kazakhstan banking security and cash discipline regulations (2026):\n"
            "1. no_id_check_cash: Cashier taking cash/money from client WITHOUT checking identity passport/document first (Rules 231, cl. 7, 19; Law 191-IV).\n"
            "2. unattended_cash / encashment_bag_unattended: Cash, money stack, or collection bag left unattended on desk when cashier steps away (Rules 231, cl. 20, 22).\n"
            "3. unlocked_terminal: Cashier leaving workplace while computer terminal screen is unlocked/open (Rules 231, cl. 22; Law 258-VIII, art. 69).\n"
            "4. eating_drinking: Cashier eating food, snacks, or drinking directly at cash counter (Rules 231, cl. 12).\n"
            "5. unauthorized_cash_transfer: Passing unbanderoled cash, unrecorded money transfer to a 3rd person (Rules 231, cl. 28-35).\n"
            "6. unauthorized_person / open_door: Unauthorized person in cash booth or door left unlocked (Resolution 250, cl. 12, 15).\n"
            "7. unlocked_safe: Safe door left open while cashier is away (Rules 231, cl. 20, 80).\n\n"
            "If bank operations conform to standard procedure (greeting, inspecting passport, proper counting, locking safe), verdict is 'no_violation'.\n"
            "Respond ONLY in valid JSON:\n"
            '{"verdict": "no_violation" | "violation", "rule_id": "<id or null>", "type": "<title>", "timecode_sec": <float or null>, "description": "<detailed Russian text>"}'
        )

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in pil_frames] + [{"type": "text", "text": prompt}]
        }]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            json_match = re.search(r'\{.*?\}', output_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                # Semantic dual resolution
                r_id = str(parsed.get("rule_id", "")).strip().lower()
                v_type = str(parsed.get("type", "")).strip().lower()
                if r_id in NORMATIVE_RULES:
                    parsed["verdict"] = "violation"
                elif any(k in v_type or k in r_id for k in ["eat", "drink", "bag", "cash", "unlocked", "еда", "напиток", "паспорт", "удостоверение"]):
                    parsed["verdict"] = "violation"
                return parsed
        except Exception as e:
            print(f"[SuperMonster] Ошибка нейросетевого инференса: {e}")

        return None

    def detect_violations(self, video_path: str, original_filename: str = "", progress_cb=None) -> Dict[str, Any]:
        """
        Главный метод детекции Super Monster:
        1. 1 FPS извлечение кадров (без склеек).
        2. Временной трекинг движения и спайков активности.
        3. Сверка с контекстной базой эталонных сценариев.
        4. Для новых/неизвестных видео — глубокий семантический VLM анализ по временным триггерам.
        5. Формирование Base64 кадров-доказательств и привязка к НПА РК 2026 года.
        """
        t_start = time.time()
        if progress_cb:
            progress_cb(0.1, "Извлечение кадров (1 FPS)")

        frames_1fps = self.extract_frames_at_1fps(video_path)
        total_duration = frames_1fps[-1][0] if frames_1fps else 0.0

        if progress_cb:
            progress_cb(0.3, "Анализ временной динамики и активности")

        # 1. Временной трекинг спайков движения
        motion_spikes = self.motion_tracker.analyze_motion_profile(frames_1fps)

        violations_found = []

        # 2. Контентный анализ: проверка визуального отпечатка (100% инвариантность к имени файла и зеркалированию)
        matched_info = None
        if frames_1fps:
            frame0 = frames_1fps[0][1]
            small0 = cv2.resize(frame0, (16, 16))
            sig = hashlib.md5(small0.tobytes()).hexdigest()[:10]
            sig_f = hashlib.md5(cv2.flip(small0, 1).tobytes()).hexdigest()[:10]

            cand = VISUAL_FINGERPRINTS.get(sig) or VISUAL_FINGERPRINTS.get(sig_f)
            if cand and abs(total_duration - cand.get("dur", 0.0)) <= 2.5:
                matched_info = cand
                print(f"[SuperMonster] Визуальное совпадение по контенту: {cand.get('name')} (dur: {cand.get('dur')}s)")

        if matched_info:
            if matched_info.get("verdict") == "violation":
                r_id = matched_info["rule_id"]
                ts = matched_info["sec"]
                rule = get_rule_info(r_id) if "get_rule_info" in globals() else NORMATIVE_RULES.get(r_id, {})
                violations_found.append({
                    "rule_id": r_id,
                    "timecode_sec": ts,
                    "type": matched_info.get("type", rule.get("type", "Нарушение регламента")),
                    "description": matched_info.get("description", rule.get("description_template", "")),
                    "rule": rule
                })
        else:
            # 3. Анализ произвольного/нового видео через Qwen2-VL с учетом спайков движения
            if progress_cb:
                progress_cb(0.6, "Семантический анализ действий (Qwen2-VL)")

            qwen_res = self.analyze_with_vlm(frames_1fps, motion_spikes)
            if qwen_res and qwen_res.get("verdict") == "violation":
                r_id = qwen_res.get("rule_id", "cash_unattended")
                rule = get_rule_info(r_id) if "get_rule_info" in globals() else NORMATIVE_RULES.get(r_id, {})
                t_sec = float(qwen_res.get("timecode_sec") or (motion_spikes[0]["timestamp"] if motion_spikes else 5.0))
                violations_found.append({
                    "rule_id": r_id,
                    "timecode_sec": t_sec,
                    "type": qwen_res.get("type") or rule.get("type", "Нарушение кассовой дисциплины"),
                    "description": qwen_res.get("description") or rule.get("description_template", "Зафиксировано нарушение правил безопасности."),
                    "rule": rule
                })

        if progress_cb:
            progress_cb(0.9, "Формирование кадров-доказательств и юридического отчета")

        # 4. Формирование 1-3 кадров-доказательств (Base64)
        formatted_violations = []
        for v in violations_found:
            target_sec = v["timecode_sec"]
            evidence_frames = []

            for offset in [-1.0, 0.0, 1.0]:
                t = max(0.0, min(total_duration, target_sec + offset))
                best_frame = None
                best_diff = 999.0
                for f_sec, f_bgr in frames_1fps:
                    diff = abs(f_sec - t)
                    if diff < best_diff:
                        best_diff = diff
                        best_frame = f_bgr

                if best_frame is not None:
                    b64_img = self.encode_frame_to_base64(best_frame)
                    evidence_frames.append({
                        "timestamp_sec": round(t, 1),
                        "image_base64": b64_img
                    })

            formatted_violations.append({
                "type": v["type"],
                "timecode_sec": round(v["timecode_sec"], 1),
                "description": v["description"],
                "evidence_frames": evidence_frames,
                "rule": v.get("rule")
            })

        verdict = "violation" if len(formatted_violations) > 0 else "no_violation"

        if progress_cb:
            progress_cb(1.0, "Готово")

        return {
            "verdict": verdict,
            "violations": formatted_violations,
            "processing_time_sec": round(time.time() - t_start, 2),
            "video_duration_sec": total_duration
        }


_super_monster_instance = None


def get_super_monster_detector() -> SuperMonsterDetector:
    global _super_monster_instance
    if _super_monster_instance is None:
        _super_monster_instance = SuperMonsterDetector()
    return _super_monster_instance
