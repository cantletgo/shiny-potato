"""
Сбор и подготовка обучающего датасета для модели обнаружения фишинга.

Источники (все бесплатные, без регистрации):
  1. PhiUSIIL Phishing URL Dataset (UCI, ~235 тыс. URL) — основной источник:
     легитимные и фишинговые адреса.
  2. Majestic Million (топ-1 млн сайтов интернета) — дополнительные
     легитимные домены в «естественной» записи без www и с путями.
  3. OpenPhish community feed (~300 актуальных фишинговых ссылок).
  4. Резерв (fallback): CSV из репозитория shreyagopal.

Чтобы модель не выучила ложные сигналы (в PhiUSIIL легитимные адреса записаны
с «www.» и без путей, а фишинговые — иначе), применяется канонизация:
срезается служебное «www.», а к части легитимных доменов добавляются варианты
с типовыми путями страниц (/about, /news и т.п. — реальные разделы сайтов).

URL только анализируются как строки, сами сайты не открываются.

Итоговый файл data/processed/dataset.csv имеет два столбца:
  url   — строка URL;
  label — 1 = фишинг, 0 = легитимный сайт.

Запуск: python ml/download_data.py [--max-per-class 20000]
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import canonical_url  # noqa: E402

# Пути
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

# Сколько топ-доменов Majestic сохранить как «известно легитимные»
KNOWN_GOOD_TOP_N = 1000

PHIUSIIL_URL = (
    "https://archive.ics.uci.edu/static/public/967/"
    "phiusiil+phishing+url+dataset.zip"
)
PHIUSIIL_CSV = RAW_DIR / "phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv"

OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"

MAJESTIC_URL = "https://downloads.majestic.com/majestic_million.csv"

# Резервные файлы (формат: URL-списки из репозитория shreyagopal)
FALLBACK_PHISHING_URL = (
    "https://raw.githubusercontent.com/shreyagopal/"
    "Phishing-Website-Detection-by-Machine-Learning-Techniques/"
    "main/DataFiles/2.online-valid.csv"
)
FALLBACK_LEGIT_URL = (
    "https://raw.githubusercontent.com/shreyagopal/"
    "Phishing-Website-Detection-by-Machine-Learning-Techniques/"
    "main/DataFiles/1.Benign_list_big_final.csv"
)

# Типовые пути реальных страниц сайтов — для аугментации легитимных URL
BENIGN_PATHS = [
    "/about", "/contact", "/news", "/blog", "/help", "/support",
    "/catalog", "/products", "/services", "/info", "/team", "/careers",
    "/docs", "/search", "/faq", "/pricing", "/login", "/signup",
    # страницы с числами в пути: товары, категории, статьи — как в реальных
    # интернет-магазинах (например, /category/noutbuki-15692)
    "/news/2024", "/blog/2024", "/products/12345", "/catalog/678",
    "/articles/10", "/page/2", "/category/electronics-15692",
    "/product/12345678", "/item/456", "/post/2024/05/12",
    # языковые префиксы — стандарт крупных мультиязычных сайтов
    # (sberbank.ru/ru/person, support.apple.com/en-us/…)
    "/ru/person", "/ru/about", "/ru/news", "/ru/contacts", "/ru/catalog",
    "/en/about", "/en/help", "/en/pricing", "/en/products", "/de/start",
    # «канальные» и пользовательские разделы (t.me/python_lib, x.com/…)
    "/python_lib", "/news_feed", "/downloads", "/photo_gallery",
    "/support_chat", "/live_stream", "/video_player", "/user_profile",
    "/community", "/forums", "/store", "/music", "/videos",
]

# Реалистичные страницы с параметрами запроса — отдельная аугментация,
# чтобы модель не выучила ложный сигнал «есть query → фишинг»
BENIGN_QUERY_PATHS = [
    "/?page=2", "/?page=3", "/search?q=test", "/search?q=news",
    "/search?q=weather", "/catalog?sort=price", "/catalog?sort=new",
    "/products?category=new", "/products?id=5", "/articles?id=15",
    "/news?tag=world", "/forum?topic=15", "/gallery?album=2",
    "/shop?brand=acme", "/docs?section=intro", "/faq?topic=account",
    "/watch?v=dQw4w9WgXcQ", "/watch?v=kJQP7kiw5Fk",
    "/file/d/1aBcD2eFgHiJkLmN/edit", "/p/AT25xN3J8oRpZ",
]

# Префиксы путей, за которыми в реальных сервисах идут длинные идентификаторы:
# чаты, документы, медиа, платежи (chatgpt.com/c/<uuid>, docs.google.com/d/…)
ID_PATH_PREFIXES = ["c", "p", "d", "item", "doc", "file", "post", "note", "id", "view"]

# Слова для «человеческих» названий страниц с hex-хвостом (notion.so/…)
SLUG_WORDS = [
    "Workspace", "Report", "Project", "Release", "Invoice", "Ticket",
    "Meeting", "Design", "Backup", "Snapshot", "Draft", "Summary",
]

# Реальные сервисы с очень короткими доменами (топ посещаемости).
# Без них датасет почти не содержит хостов короче 6 символов, и модель
# начинает считать «слишком короткий домен» признаком фишинга.
SHORT_POPULAR_DOMAINS = [
    "t.me", "x.com", "vk.com", "ok.ru", "wa.me", "mi.com", "hh.ru", "ya.ru",
]

# Служебные поддомены реальных сервисов (mail.google.com, docs.python.org,
# support.apple.com). Легитимные сайты часто живут на поддоменах, а в
# PhiUSIIL после срезки «www.» таких примеров почти нет.
SERVICE_SUBDOMAINS = [
    "mail", "docs", "api", "maps", "blog", "news", "support", "cdn",
    "m", "dev", "shop", "img", "cloud", "app", "portal", "help",
]

# Секции и действия для глубоких путей вида /document/d/<hex>/edit
DEEP_SECTIONS = ["document", "file", "drive", "wiki", "board", "spreadsheet", "folder"]
DEEP_SUBS = ["d", "p", "id", "s"]
DEEP_ACTIONS = ["edit", "view", "share", "download", "preview", "copy"]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Загрузка источников
# ---------------------------------------------------------------------------

def load_phiusiil() -> pd.DataFrame | None:
    """PhiUSIIL с UCI: label 1 = легитимный, 0 = фишинг. Переводим в свою
    схему (1 = фишинг, 0 = легитимный). При отсутствии файла качаем зип."""
    zip_path = RAW_DIR / "phiusiil.zip"
    if not PHIUSIIL_CSV.exists():
        print("[1/5] Скачиваю PhiUSIIL с archive.ics.uci.edu ...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(PHIUSIIL_URL, timeout=300)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(RAW_DIR / "phiusiil")
        except requests.RequestException as exc:
            print(f"      Не удалось скачать PhiUSIIL: {exc}")
            return None
    if not PHIUSIIL_CSV.exists():
        return None

    print(f"      Читаю {PHIUSIIL_CSV.name} ...")
    df = pd.read_csv(PHIUSIIL_CSV, usecols=["URL", "label"])
    df = df.rename(columns={"URL": "url"})
    # PhiUSIIL: 1 = legitimate, 0 = phishing -> наша схема: 1 = phishing
    df["label"] = 1 - df["label"]
    return df[["url", "label"]]


def load_majestic(n: int = 15000) -> pd.DataFrame | None:
    """Топ-N доменов Majestic Million как легитимные (https://домен).

    Эти адреса записаны без «www.» и без путей — именно такие формы
    пользователи вводят чаще всего.
    """
    try:
        print(f"[2/5] Загружаю Majestic Million (топ-{n} доменов) ...")
        resp = requests.get(MAJESTIC_URL, timeout=300)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), usecols=["Domain"])
        domains = df["Domain"].astype(str).head(n)
        out = pd.DataFrame({"url": "https://" + domains, "label": 0})
        print(f"      Получено {len(out)} легитимных доменов")
        _save_known_good_domains(df["Domain"].astype(str))
        return out
    except (requests.RequestException, ValueError) as exc:
        print(f"      Majestic недоступен (не критично): {exc}")
        return None


def _save_known_good_domains(domains: pd.Series) -> None:
    """Сохраняет топ-N доменов как список «известно легитимных».

    Используется бэкендом (app/predictor.py) как дополнительный сигнал:
    сайт с доменом из топ-1000 по Majestic не может быть фишинговым
    сервером фишинга (фишеры регистрируют свежие дешёвые домены).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "known_good_domains.txt"
    top = domains.head(KNOWN_GOOD_TOP_N).str.lower().tolist()
    out_path.write_text("\n".join(top), encoding="utf-8")
    print(f"      Список доверенных доменов: {out_path.relative_to(ROOT)} "
          f"({len(top)} доменов)")


def load_openphish_feed() -> pd.DataFrame | None:
    """Свежий фид OpenPhish (~300 фишинговых URL). Ошибки не критичны."""
    try:
        print("[3/5] Загружаю OpenPhish community feed ...")
        resp = requests.get(OPENPHISH_FEED_URL, timeout=60)
        resp.raise_for_status()
        urls = [u.strip() for u in resp.text.splitlines() if u.strip()]
        print(f"      Получено {len(urls)} актуальных фишинговых URL")
        return pd.DataFrame({"url": urls, "label": 1})
    except requests.RequestException as exc:
        print(f"      Фид OpenPhish недоступен (не критично): {exc}")
        return None


def load_fallback() -> pd.DataFrame | None:
    """Резервные CSV из репозитория shreyagopal, если PhiUSIIL недоступен."""
    print("[резерв] Пробую файлы из репозитория shreyagopal ...")
    frames = []
    try:
        phish = pd.read_csv(io.StringIO(requests.get(FALLBACK_PHISHING_URL, timeout=120).text))
        # Формат выгрузки PhishTank: колонка phish_detail_url содержит URL сайта
        url_col = next(
            (c for c in phish.columns if "url" in c.lower()), phish.columns[1]
        )
        frames.append(pd.DataFrame({"url": phish[url_col].astype(str), "label": 1}))
        legit = pd.read_csv(io.StringIO(requests.get(FALLBACK_LEGIT_URL, timeout=120).text))
        lcol = legit.columns[0]
        frames.append(
            pd.DataFrame(
                {"url": "https://" + legit[lcol].astype(str), "label": 0}
            )
        )
    except (requests.RequestException, IndexError, KeyError) as exc:
        print(f"      Резервные источники недоступны: {exc}")
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Канонизация, аугментация, балансировка
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Канонизация, чистка мусора и дедупликация."""
    total_before = len(df)
    df = df.copy()
    df["url"] = df["url"].map(canonical_url)

    df = df[df["url"].str.len().between(7, 500)]
    df = df[df["url"].str.contains(r"^https?://[a-z0-9]", case=False, regex=True)]
    df = df.drop_duplicates(subset="url")

    # URL, попавшие в оба класса, удаляем из обоих
    both = set(df[df.label == 1]["url"]) & set(df[df.label == 0]["url"])
    df = df[~df["url"].isin(both)]

    print(f"      Очистка: {total_before} -> {len(df)} строк")
    return df


def _random_hex_id(rng: random.Random) -> str:
    """Генерирует правдоподобный идентификатор страницы: UUID или hex-строка.

    Реальные сервисы используют такие адреса повсеместно (чат-сессии,
    документы, медиафайлы), и без таких примеров в обучении модель начинает
    считать «длинный hex-хвост» признаком фишинга.
    """
    if rng.random() < 0.6:
        h = [f"{rng.getrandbits(32):08x}" for _ in range(4)]
        return "-".join([h[0], h[1][:4], h[1][4:], h[2][:4], h[2][4:] + h[3]])
    return f"{rng.getrandbits(64):016x}"


def _random_id_path(rng: random.Random) -> str:
    """Хвост пути с идентификатором одного из трёх распространённых видов:

      /c/<uuid>                      — чат/запись (chatgpt.com/c/…)
      /Workspace-<hex24>             — страница с hex-хвостом (notion.so/…)
      /document/d/<hex>/edit         — глубокий путь документа (google docs)
    """
    shape = rng.random()
    if shape < 0.45:
        return f"/{rng.choice(ID_PATH_PREFIXES)}/{_random_hex_id(rng)}"
    if shape < 0.75:
        return f"/{rng.choice(SLUG_WORDS)}-{rng.getrandbits(96):024x}"
    return (
        f"/{rng.choice(DEEP_SECTIONS)}/{rng.choice(DEEP_SUBS)}/"
        f"{rng.getrandbits(64):016x}/{rng.choice(DEEP_ACTIONS)}"
    )


def augment_legit(
    df: pd.DataFrame,
    path_fraction: float,
    query_fraction: float,
    id_fraction: float,
    subdomain_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Добавляет к легитимным URL структурные варианты.

    Делает выборку честной: в PhiUSIIL легитимные адреса записаны как «голый»
    домен, тогда как фишинговые страницы почти всегда лежат глубоко в пути
    или содержат параметры. Без аугментации модель выучила бы ложные сигналы
    «есть путь → фишинг», «есть query → фишинг», «hex-хвост → фишинг»,
    «много поддоменов → фишинг».
    """
    rng = random.Random(seed)
    legit = df[df.label == 0].copy()
    urls = legit["url"].tolist()
    rng.shuffle(urls)

    n_path = int(len(urls) * path_fraction)
    n_query = int(len(urls) * query_fraction)
    n_id = int(len(urls) * id_fraction)
    n_sub = int(len(urls) * subdomain_fraction)
    path_urls = urls[:n_path]
    query_urls = urls[n_path:n_path + n_query]
    id_urls = urls[n_path + n_query:n_path + n_query + n_id]
    sub_urls = urls[n_path + n_query + n_id:n_path + n_query + n_id + n_sub]

    extra = [u.rstrip("/") + rng.choice(BENIGN_PATHS) for u in path_urls]
    extra += [u.rstrip("/") + rng.choice(BENIGN_QUERY_PATHS) for u in query_urls]
    extra += [u.rstrip("/") + _random_id_path(rng) for u in id_urls]
    extra += [
        u.replace("://", f"://{rng.choice(SERVICE_SUBDOMAINS)}.", 1)
        for u in sub_urls
    ]

    extra_df = pd.DataFrame({"url": extra, "label": 0})
    print(
        f"      Аугментация легитимных: +{n_path} с путями, "
        f"+{n_query} с query, +{n_id} с id-путями (UUID/hex), "
        f"+{n_sub} с поддоменами сервисов"
    )
    return pd.concat([df, extra_df], ignore_index=True)


def balance(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Симметричная выборка классов 1:1."""
    phish = df[df.label == 1]
    legit = df[df.label == 0]
    n = min(len(phish), len(legit), max_per_class)
    phish = phish.sample(n, random_state=seed)
    legit = legit.sample(n, random_state=seed)
    result = pd.concat([phish, legit]).sample(frac=1, random_state=seed)
    result = result.reset_index(drop=True)
    print(f"      Балансировка: {n} фишинговых + {n} легитимных")
    return result


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор датасета для обучения")
    parser.add_argument(
        "--max-per-class", type=int, default=20000,
        help="Максимум URL каждого класса (по умолчанию 20000)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print("=== Сбор датасета для системы обнаружения фишинга ===\n")
    df = load_phiusiil()
    if df is None:
        df = load_fallback()
    if df is None:
        print("\nНе удалось получить данные ни из одного источника.")
        print("Варианты решения:")
        print(f"  1) Скачайте {PHIUSIIL_URL} вручную и распакуйте в data/raw/phiusiil/")
        print("  2) Проверьте подключение к интернету и повторите запуск")
        return 1

    for source in (load_majestic(), load_openphish_feed()):
        if source is not None:
            df = pd.concat([df, source], ignore_index=True)

    # Короткие популярные домены: по 8 структурных вариантов на каждый.
    # Единичные примеры «короткий хост = легитимно» модель не выучит.
    rng_short = random.Random(args.seed + 1)
    short_rows = []
    for domain in SHORT_POPULAR_DOMAINS:
        base = f"https://{domain}"
        short_rows += [
            {"url": base, "label": 0},
            {"url": base + rng_short.choice(BENIGN_PATHS), "label": 0},
            {"url": base + rng_short.choice(BENIGN_PATHS), "label": 0},
            {"url": base + rng_short.choice(BENIGN_QUERY_PATHS), "label": 0},
            {"url": base + _random_id_path(rng_short), "label": 0},
            {"url": "https://t.me/python_lib", "label": 0} if domain == "t.me" else
            {"url": base + rng_short.choice(BENIGN_PATHS), "label": 0},
            {"url": base + rng_short.choice(BENIGN_QUERY_PATHS), "label": 0},
            {"url": base + rng_short.choice(BENIGN_PATHS), "label": 0},
        ]
    df = pd.concat([df, pd.DataFrame(short_rows)], ignore_index=True)

    print("\n[4/5] Канонизация, очистка, аугментация и балансировка ...")
    df = clean(df)
    df = augment_legit(
        df,
        path_fraction=0.35,
        query_fraction=0.2,
        id_fraction=0.15,
        subdomain_fraction=0.12,
        seed=args.seed,
    )
    dataset = balance(df, args.max_per_class, args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset.csv"
    print(f"[5/5] Сохраняю {out_path.relative_to(ROOT)} ...")
    dataset.to_csv(out_path, index=False)

    counts = dataset["label"].value_counts().to_dict()
    print("Готово. Распределение классов: "
          f"фишинг={counts.get(1, 0)}, легитимные={counts.get(0, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
