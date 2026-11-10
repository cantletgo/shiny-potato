"""
Загрузка обученной модели и предсказание вердикта для URL.

Класс Predictor — синглтон: модель загружается один раз при старте
приложения (см. app/main.py) и переиспользуется для всех запросов.
Признаки извлекаются тем же модулем, что и при обучении (ml/features.py),
поэтому распределение признаков на входе модели совпадает с обучающим.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
KNOWN_GOOD_PATH = ARTIFACTS_DIR / "known_good_domains.txt"

# Модуль признаков живёт в ml/, подключаем его к пути приложения
sys.path.insert(0, str(ROOT / "ml"))
from features import (  # noqa: E402
    SHORTENERS,
    canonical_url,
    extract_features,
    explain_features,
    feature_table,
)


class Predictor:
    """Обёртка над моделью: загрузка артефактов и предсказание."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Модель не найдена: {MODEL_PATH}\n"
                "Сначала обучите её: python ml/train.py"
            )
        self.model = joblib.load(MODEL_PATH)
        self.info: Dict = {}
        if METRICS_PATH.exists():
            self.info = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        # Список «известно легитимных» доменов (топ Majestic Million).
        # Файл создаётся ml/download_data.py; при отсутствии проверка
        # просто не применяется и вердикт определяет только модель.
        self.known_good_domains: set[str] = set()
        if KNOWN_GOOD_PATH.exists():
            self.known_good_domains = {
                line.strip().lower()
                for line in KNOWN_GOOD_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

    @property
    def feature_names(self):
        return self.model.feature_names_in_

    def _is_known_good(self, host: str) -> bool:
        """Домен из топ Majestic (полное имя или домен 2-го уровня)."""
        if not self.known_good_domains:
            return False
        labels = [lb for lb in host.split(".") if lb]
        registrable = ".".join(labels[-2:]) if len(labels) >= 2 else host
        return host in self.known_good_domains or registrable in self.known_good_domains

    def check(self, url: str) -> Dict:
        """Полный расчёт по одному URL: вердикт, уверенность, объяснения."""
        canonical = canonical_url(url)
        feats = extract_features(canonical)
        X = pd.DataFrame([[feats[name] for name in self.feature_names]],
                         columns=self.feature_names)

        proba_phishing = float(self.model.predict_proba(X)[0][1])

        # Дополнительный сигнал: домен входит в топ Majestic Million.
        # Фишинговые сайты живут на свежих дешёвых доменах и в топ-1000
        # самых посещаемых сайтов не встречаются, поэтому такой домен
        # считается заведомо легитимным независимо от оценки модели.
        parsed = urlparse(canonical)
        host = parsed.netloc.split(":")[0].split("@")[-1]
        # Сокращатели ссылок исключаем из доверенных, даже если домен в топе:
        # они скрывают настоящий адрес назначения
        trusted = self._is_known_good(host) and host not in SHORTENERS
        if trusted:
            proba_phishing = min(proba_phishing, 0.05)

        label = 1 if proba_phishing >= 0.5 else 0
        verdict = "phishing" if label == 1 else "safe"

        explanations = explain_features(canonical, feats)
        if trusted:
            explanations.insert(
                0,
                "Домен входит в топ-1000 самых посещаемых сайтов интернета "
                "(Majestic Million) — известный легитимный домен",
            )

        return {
            "url": url,
            "verdict": verdict,
            "label": label,
            "trusted": trusted,
            "confidence": round(max(proba_phishing, 1 - proba_phishing), 4),
            "risk_score": round(proba_phishing * 100, 1),
            "probabilities": {
                "phishing": round(proba_phishing, 4),
                "safe": round(1 - proba_phishing, 4),
            },
            "explanations": explanations,
            "features": feature_table(feats),
            "model": {
                "class": self.info.get("model_class", type(self.model).__name__),
                "trained_at_utc": self.info.get("trained_at_utc", ""),
                "f1": str(self.info.get("metrics", {}).get("f1", "")),
            },
        }

    def model_info(self) -> Dict:
        """Данные о модели для GET /api/model/info."""
        if not self.info:
            return {
                "model_class": type(self.model).__name__,
                "trained_at_utc": "",
                "dataset": {},
                "metrics": {},
                "confusion_matrix": [],
                "feature_importances": [],
                "features_used": list(self.feature_names),
            }
        return {
            "model_class": self.info.get("model_class"),
            "trained_at_utc": self.info.get("trained_at_utc"),
            "dataset": self.info.get("dataset", {}),
            "metrics": self.info.get("metrics", {}),
            "confusion_matrix": self.info.get("confusion_matrix", []),
            "feature_importances": self.info.get("feature_importances", []),
            "features_used": self.info.get("feature_names", []),
        }


# Единственный экземпляр на процесс приложения
predictor = Predictor()
