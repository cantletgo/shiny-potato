"""
Обучение модели обнаружения фишинговых сайтов.

Пайплайн:
  1. Загрузка датасета (data/processed/dataset.csv) и извлечение признаков
     (с кэшированием — см. ml/dataset.py).
  2. Разделение на обучающую/тестовую выборки (80/20, стратифицированно).
  3. Обучение RandomForestClassifier.
  4. Расчёт метрик: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.
  5. Сохранение артефактов: ml/artifacts/model.joblib и metrics.json.

Запуск: python ml/train.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DATASET_PATH, load_dataset  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": 42,
}


def main() -> int:
    print("=== Обучение модели обнаружения фишинга ===\n")

    X, y, _ = load_dataset(DATASET_PATH)
    print(f"Признаков: {len(FEATURE_NAMES)}; всего примеров: {len(y)}")
    print(f"  фишинг: {int((y == 1).sum())}, легитимные: {int((y == 0).sum())}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Обучающая выборка: {len(X_train)}, тестовая: {len(X_test)}")

    model = RandomForestClassifier(**MODEL_PARAMS)
    print(f"\nОбучаю {type(model).__name__} (n_estimators={MODEL_PARAMS['n_estimators']}) ...")
    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = round(time.time() - start, 1)
    print(f"Обучение заняло {train_seconds} c")

    print("\nОцениваю на тестовой выборке ...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Метрики (тест 20%) ---")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value}")
    print("\nМатрица ошибок (строки — истина, столбцы — прогноз):")
    print(f"  Легитимные: TN={cm[0][0]}, ошибочно flagged={cm[0][1]}")
    print(f"  Фишинг:     пропущено={cm[1][0]}, TP={cm[1][1]}")

    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    print("\nТоп-10 важных признаков:")
    for name, imp in importances[:10]:
        print(f"  {name:<24} {imp:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(model, model_path)

    info = {
        "model_class": type(model).__name__,
        "model_params": MODEL_PARAMS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_seconds": train_seconds,
        "dataset": {
            "source": "PhiUSIIL (UCI) + OpenPhish community feed",
            "path": str(DATASET_PATH.relative_to(ROOT)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_phishing": int((y == 1).sum()),
            "n_legit": int((y == 0).sum()),
        },
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "feature_names": FEATURE_NAMES,
        "feature_importances": [
            {"name": n, "importance": round(float(i), 4)} for n, i in importances
        ],
    }
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nСохранено:\n  модель:  {model_path.relative_to(ROOT)}")
    print(f"  метрики: {metrics_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
