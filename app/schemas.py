"""Pydantic-схемы запросов и ответов API проверки URL."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    """Тело запроса на проверку URL."""

    url: str = Field(
        ...,
        min_length=4,
        max_length=2000,
        description="Проверяемый URL, например https://example.com/page",
    )


class FeatureItem(BaseModel):
    """Одна строка таблицы признаков."""

    name: str
    description: str
    value: float


class CheckResponse(BaseModel):
    """Результат проверки URL."""

    url: str
    verdict: str = Field(..., description="«phishing» или «safe»")
    label: int = Field(..., description="1 = фишинг, 0 = легитимный")
    trusted: bool = Field(
        False,
        description="Домен входит в список топ-1000 сайтов (Majestic Million)",
    )
    confidence: float = Field(..., description="Уверенность модели, 0..1")
    risk_score: float = Field(..., description="Риск фишинга, 0..100")
    probabilities: Dict[str, float]
    explanations: List[str] = Field(..., description="Почему модель так решила")
    features: List[FeatureItem]
    model: Dict[str, str]


class FeatureImportanceItem(BaseModel):
    """Важность одного признака в обученной модели."""

    name: str
    importance: float


class ModelInfoResponse(BaseModel):
    """Сведения о текущей модели (для отчёта и страницы «О модели»)."""

    model_class: str
    trained_at_utc: str
    dataset: Dict
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    feature_importances: List[FeatureImportanceItem]
    features_used: List[str]
