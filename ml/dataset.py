"""
Общие утилиты загрузки датасета с извлечением признаков.

Используется ml/train.py и ml/evaluate.py. Чтобы не извлекать признаки заново
при каждом запуске, матрица признаков кэшируется рядом с датасетом и
инвалидируется по контрольной сумме файла датасета.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "processed" / "dataset.csv"
CACHE_PATH = ROOT / "data" / "processed" / "features_cache.csv"


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(dataset_path: Path = DATASET_PATH, use_cache: bool = True):
    """Загружает датасет и возвращает (X, y, df).

    X — DataFrame с признаками (FEATURE_NAMES), y — метки классов
    (1 = фишинг, 0 = легитимный). Признаки кэшируются: если файл кэша
    соответствует текущей версии датасета (md5 в первой строке кэша),
    извлечение признаков пропускается.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {dataset_path}\n"
            "Сначала запустите: python ml/download_data.py"
        )

    md5 = _file_md5(dataset_path)
    if use_cache and CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached_md5 = f.readline().strip()
            if cached_md5 == md5:
                print(f"Использую кэш признаков: {CACHE_PATH.name}")
                cached = pd.read_csv(CACHE_PATH, skiprows=1)
                return cached[FEATURE_NAMES], cached["label"], cached[["url", "label"]]
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print("Кэш признаков повреждён — пересчитываю заново")

    print(f"Извлекаю признаки из {len(pd.read_csv(dataset_path))} URL ...")
    df = pd.read_csv(dataset_path)
    feats = pd.DataFrame(
        (extract_features(u) for u in df["url"]), columns=FEATURE_NAMES
    )
    feats.insert(0, "url", df["url"].values)
    feats["label"] = df["label"].values

    if use_cache:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(md5 + "\n")
            feats.to_csv(f, index=False)
        print(f"Кэш признаков сохранён: {CACHE_PATH.name}")

    return feats[FEATURE_NAMES], feats["label"], feats[["url", "label"]]
