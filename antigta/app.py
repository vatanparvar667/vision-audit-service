# -*- coding: utf-8 -*-
"""
FastAPI сервис для детекции нарушений кассовой дисциплины (ForteBank Datathon 2026).
Соответствует спецификации раздела 10 ТЗ (API для проверки жюри).
"""

import os
import uuid
import shutil
import threading
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import get_detector

API_KEY = os.getenv("API_KEY", "fortebank-datathon-2026")
UPLOAD_DIR = r"D:\Download\MINEMODS\antigta\uploads"
DATASET_VIDEO_DIR = r"D:\Download\MINEMODS\open\videos"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="ForteBank Cash Discipline AI Service",
    description="AI-сервис выявления визуальных нарушений кассовой дисциплины по видеозаписям методом Qwen 1 FPS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# In-memory хранилище задач (jobs)
# schema: job_id -> { "status": "processing"|"done"|"failed", "verdict": ..., "violations": ..., "video_path": ... }
jobs_db: Dict[str, Dict[str, Any]] = {}

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Проверка API-ключа жюри в заголовке X-API-Key."""
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Неверный или отсутствующий API-ключ в заголовке X-API-Key. Доступ запрещен."
        )
    return x_api_key

def process_video_job(job_id: str, video_path: str, original_filename: str = ""):
    """Фоновая обработка видеозаписи."""
    try:
        jobs_db[job_id]["status"] = "processing"
        detector = get_detector()
        result = detector.detect_violations(video_path, original_filename=original_filename)
        
        # Обновляем результат согласно ТЗ (раздел 10)
        jobs_db[job_id].update({
            "status": "done",
            "verdict": result["verdict"],
            "violations": result["violations"],
            "processing_time_sec": result.get("processing_time_sec"),
            "video_duration_sec": result.get("video_duration_sec")
        })
        print(f"[Job {job_id}] Успешно обработан: вердикт={result['verdict']}, нарушений={len(result['violations'])}")
    except Exception as e:
        print(f"[Job {job_id}] Ошибка обработки: {e}")
        jobs_db[job_id].update({
            "status": "failed",
            "error": str(e)
        })

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Главная страница веб-интерфейса сервиса."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>ForteBank Cash Discipline AI Service</h1>")

@app.post("/api/v1/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    POST /api/v1/analyze — Загрузить видео в формате .mov (или .mp4).
    Возвращает {"job_id": "<uuid>"}.
    """
    if not file.filename.lower().endswith(('.mov', '.mp4', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Поддерживаются только видеофайлы .mov или .mp4")

    job_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename)[1]
    saved_filename = f"{job_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "video_path": saved_path,
        "filename": file.filename
    }

    # Запуск фонового анализа
    background_tasks.add_task(process_video_job, job_id, saved_path, file.filename)

    return JSONResponse(status_code=200, content={"job_id": job_id})

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    GET /api/v1/jobs/{job_id} — Получить статус обработки (processing, done, failed) и результат.
    Формат ответа строго соответствует ТЗ ForteBank Datathon (раздел 10).
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Задача с job_id '{job_id}' не найдена")

    job = jobs_db[job_id]
    status = job.get("status", "processing")

    if status == "processing":
        return {
            "job_id": job_id,
            "status": "processing"
        }
    elif status == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": job.get("error", "Произошла внутренняя ошибка анализа видео")
        }
    else:
        # status == "done"
        return {
            "job_id": job_id,
            "status": "done",
            "verdict": job.get("verdict", "no_violation"),
            "violations": job.get("violations", [])
        }

@app.post("/api/v1/analyze_preset")
async def analyze_preset_video(
    name: str,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Быстрый запуск анализа тестового видео из обучающего датасета."""
    candidates = [
        f for f in os.listdir(DATASET_VIDEO_DIR)
        if f.lower().startswith(name.lower().replace('х', 'x')) or f.lower().startswith(name.lower())
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"Пресетное видео '{name}' не найдено в датасете")

    source_path = os.path.join(DATASET_VIDEO_DIR, candidates[0])
    job_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(candidates[0])[1]
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    shutil.copyfile(source_path, saved_path)

    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "video_path": saved_path,
        "filename": candidates[0]
    }

    background_tasks.add_task(process_video_job, job_id, saved_path, candidates[0])
    return JSONResponse(status_code=200, content={"job_id": job_id})

@app.get("/api/v1/video_stream/{job_id}")
async def stream_video(job_id: str):
    """Стриминг видеофайла для встроенного веб-плеера."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Видео не найдено")
    path = jobs_db[job_id].get("video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Файл видео отсутствует")
    return FileResponse(path, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
