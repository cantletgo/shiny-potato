"""
Сравнение нескольких моделей классификации на одном и том же наборе признаков.

Обучает RandomForest, LogisticRegression и GradientBoosting на одинаковых
выборках и печатает сравнительную таблицу метрик (Accuracy, Precision,
Recall, F1, ROC-AUC), матрицы ошибок и важность признаков лучшей модели.
Результаты дублируются в ml/artifacts/comparison.json — удобно вставлять
в отчёт по курсовой работе.

Запуск: python ml/evaluate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DATASET_PATH, load_dataset  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.2, random_state=42
    ),
}


def score_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - start, 1)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = {
        "model": name,
        "train_seconds": elapsed,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return result


def main() -> int:
    print("=== Сравнение моделей обнаружения фишинга ===\n")

    X, y, _ = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Обучающая выборка: {len(X_train)}, тестовая: {len(X_test)}\n")

    results = []
    for name, model in MODELS.items():
        print(f"Обучаю {name} ...")
        results.append(score_model(name, model, X_train, X_test, y_train, y_test))

    header = f"{'Модель':<20}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>8}{'ROC-AUC':>9}{'время,с':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<20}{r['accuracy']:>9.4f}{r['precision']:>10.4f}"
            f"{r['recall']:>8.4f}{r['f1']:>8.4f}{r['roc_auc']:>9.4f}"
            f"{r['train_seconds']:>9.1f}"
        )

    best = max(results, key=lambda r: r["f1"])
    cm = best["confusion_matrix"]
    print(f"\nЛучшей по F1 признана модель: {best['model']}")
    print("Матрица ошибок (строки — истина, столбцы — прогноз):")
    print(f"  Легитимные: TN={cm[0][0]}, ошибочно flagged={cm[0][1]}")
    print(f"  Фишинг:     пропущено={cm[1][0]}, TP={cm[1][1]}")

    rf = MODELS["RandomForest"]
    importances = sorted(
        zip(FEATURE_NAMES, rf.feature_importances_), key=lambda kv: kv[1], reverse=True
    )
    print("\nТоп-15 важных признаков RandomForest:")
    for name, imp in importances[:15]:
        print(f"  {name:<24} {imp:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "comparison.json"
    out_path.write_text(
        json.dumps(
            {"results": results, "best_by_f1": best["model"]},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nРезультаты сохранены: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
