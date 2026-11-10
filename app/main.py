"""
Веб-система обнаружения фишинговых сайтов — бэкенд на FastAPI.

Маршруты:
  POST /api/check       — проверка URL: вердикт, уверенность, объяснения
  GET  /api/model/info  — сведения о модели и её метриках
  GET  /health          — проверка живости сервиса
  GET  /                — веб-интерфейс (app/static/index.html)
  /docs                 — автодокументация Swagger

Запуск: uvicorn app.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.predictor import predictor
from app.schemas import CheckRequest, CheckResponse, ModelInfoResponse

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "app" / "static"

app = FastAPI(
    title="Веб-система обнаружения фишинговых сайтов",
    description=(
        "API проверки URL на фишинг. Модель RandomForest обучена на признаках "
        "URL (PhiUSIIL + OpenPhish). Вердикт сопровождается объяснением: "
        "какие признаки URL сработали."
    ),
    version="1.0.0",
)


@app.post("/api/check", response_model=CheckResponse,
          summary="Проверить URL на фишинг")
def check_url(payload: CheckRequest) -> CheckResponse:
    url = payload.url.strip()
    if len(url) < 4 or "://" not in url and "." not in url:
        raise HTTPException(
            status_code=400,
            detail="Похоже, это не URL. Пример: https://example.com/page",
        )
    try:
        return CheckResponse(**predictor.check(url))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось разобрать URL: {exc}")


@app.get("/api/model/info", response_model=ModelInfoResponse,
         summary="Сведения о модели")
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**predictor.model_info())


@app.get("/health", summary="Проверка живости сервиса")
def health() -> dict:
    return {"status": "ok", "model": predictor.info.get("model_class", "?")}


# Веб-интерфейс: статика подключается последней, чтобы не перекрыть /api и /docs
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
