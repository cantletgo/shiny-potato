"""
Извлечение признаков из URL для модели обнаружения фишинга.

Этот модуль — единая точка извлечения признаков: он используется и при
обучении модели (ml/train.py), и при обслуживании запросов (app/predictor.py).
Такой подход исключает расхождение «train/serving skew» — признаки на входе
модели всегда вычисляются одинаково.

Важно: URL только анализируется как строка, сами ссылки никуда не отправляются
и не открываются.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Справочные наборы (сигналы фишинга)
# ---------------------------------------------------------------------------

# TLD, которые статистически чаще других используются фишинговыми сайтами
# (дешёвые или слабо контролируемые доменные зоны).
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",          # бесплатные зоны Freenom
    "top", "xyz", "click", "link", "work",
    "loan", "men", "party", "review", "stream",
    "country", "gdn", "mom", "xin", "icu",
    "cyou", "cam", "quest", "fit", "sbs",
}

# Названия популярных брендов, которые фишеры подделывают чаще всего.
# Сопоставление — только по точному совпадению сегмента домена (см.
# _brand_in_domain): официальный домен бренда (paypal.com, sberbank.ru)
# сигналом НЕ считается, а вот «paypal» внутри чужого домена
# (paypal-login.tk, secure.paypal.evil.tk) — признак подделки.
BRAND_KEYWORDS = [
    "paypal", "apple", "icloud", "google", "amazon", "facebook", "instagram",
    "netflix", "microsoft", "office365", "outlook", "linkedin", "wellsfargo",
    "chase", "bankofamerica", "hsbc", "dhl", "fedex", "ups", "usps",
    "sberbank", "vtb", "tinkoff", "tbank", "alfabank", "gazprombank",
    "yandex", "ozon", "wildberries", "avito", "gosuslugi", "vk", "mailru",
]

# Короткие слова, характерные для фишинговых страниц (призыв действия,
# имитация безопасности, номер регистрации и т.п.).
SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "secure", "security", "account", "update", "confirm", "webscr",
    "recover", "unlock", "suspended", "billing", "invoice", "payment",
    "wallet", "restore", "support", "authenticate", "reset",
]

# Популярные сокращатели ссылок — скрывают настоящий адрес назначения.
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "shorturl.at", "rb.gy", "rebrand.ly", "clck.ru", "bit.do",
    "tiny.cc", "s.id", "v.gd",
}

IP_ADDRESS_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-fA-F]+$")

# ---------------------------------------------------------------------------
# Описания признаков (используются фронтендом и отчётами)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "url_length": "Полная длина URL, символов",
    "hostname_length": "Длина доменного имени",
    "has_ip": "Вместо домена указан IP-адрес",
    "has_at": "Символ «@» в URL (скрытие настоящего адреса)",
    "has_redirect": "«//» внутри пути (перенаправление)",
    "has_hyphen_in_domain": "Дефис в доменном имени",
    "num_dots": "Количество точек в URL",
    "num_subdomains": "Количество уровней поддоменов",
    "is_https": "Используется HTTPS",
    "num_digits": "Количество цифр в URL",
    "digit_ratio": "Доля цифр в URL",
    "num_special_chars": "Количество специальных символов (@ ? - = & % $ ! _ ~ ; ( ) *, и т.п.)",
    "suspicious_tld": "Подозрительный домен верхнего уровня (TLD)",
    "brand_in_domain": "Брендовое слово внутри домена",
    "is_shortener": "Ссылка — сокращатель URL",
    "has_punycode": "Punycode-домен (xn--), возможная подделка символов",
    "has_port": "Явно указан нестандартный порт",
    "path_depth": "Глубина пути (число сегментов)",
    "query_length": "Длина строки запроса (после «?»)",
    "num_params": "Количество параметров запроса",
    "domain_entropy": "Энтропия домена (хаотичность символов)",
    "suspicious_words": "Количество «фишинговых» слов (login, verify, account…)",
}

FEATURE_NAMES: List[str] = list(FEATURE_DESCRIPTIONS.keys())


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _normalize(url: str) -> str:
    """Приводит URL к разбираемому виду: добавляет схему, если её нет.

    Без схемы считаем адрес https:// — пользователь обычно вводит «yandex.ru»,
    имея в виду безопасное соединение (современные браузеры тоже пробуют
    https первым).
    """
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    return url


def canonical_url(url: str) -> str:
    """Канонизация URL для обучения и обслуживания.

    Добавляет схему (если её нет), приводит имя хоста к нижнему регистру и
    срезает служебный префикс «www.». Нужна, чтобы модель не выучила ложный
    признак «есть www → сайт легитимный»: одна и та же страница с www и без
    него даёт одинаковые признаки. Используется и при сборке датасета,
    и при предсказании.
    """
    url = _normalize(url)
    parsed = urlparse(url)
    netloc = parsed.netloc

    # Отделяем userinfo (user:pass@), приводим хост к нижнему регистру
    userinfo = ""
    if "@" in netloc:
        userinfo, netloc = netloc.rsplit("@", 1)
        userinfo += "@"
    host_name, _, port_part = netloc.lower().partition(":")

    # Срезаем «www.» только если после него остаются ещё два уровня
    labels = host_name.split(".")
    if len(labels) >= 3 and labels[0] == "www":
        host_name = ".".join(labels[1:])

    new_netloc = userinfo + host_name + (":" + port_part if port_part else "")
    if new_netloc != parsed.netloc:
        url = url.replace(parsed.netloc, new_netloc, 1)
    return url


def _hostname(parsed) -> str:
    """Извлекает имя хоста без порта и userinfo (user:pass@)."""
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    host = netloc.split(":", 1)[0].lower()
    return host.rstrip(".")


def _shannon_entropy(text: str) -> float:
    """Энтропия Шеннона строки, бит/символ. Высокие значения — «случайные» домены."""
    if not text:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _brand_in_domain(host: str, second_level: str) -> int:
    """Брендовое слово в «чужом» домене — сигнал подделки.

    Официальный домен бренда (second_level == бренд: paypal.com,
    sberbank.ru, включая поддомены вида mail.google.com) — НЕ сигнал.
    Сигнал — бренд, встроенный в чужой домен:
      * целым сегментом-поддоменом: secure.paypal.evil.tk;
      * сегментом дефис-домена:     paypal-login.tk, apple-id.xyz.
    Точное сравнение сегментов исключает ложные срабатывания подстрок
    («sber» внутри «sberbank», «ups» внутри «groups»).
    """
    labels = [lb for lb in host.split(".") if lb]
    subdomain_labels = labels[:-2]
    sld_parts = second_level.split("-")
    for brand in BRAND_KEYWORDS:
        if second_level == brand:
            continue  # официальный домен бренда
        if brand in subdomain_labels or brand in sld_parts:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Главная функция: извлечение всех признаков
# ---------------------------------------------------------------------------

def extract_features(url: str) -> Dict[str, float]:
    """Вычисляет все признаки для одного URL.

    Возвращает словарь вида {имя_признака: значение}. Все значения числовые,
    поэтому пригодны для прямой подачи в sklearn-модель.
    """
    raw = canonical_url(url)
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    host = _hostname(parsed)
    path = parsed.path or ""
    query = parsed.query or ""

    labels = [lb for lb in host.split(".") if lb]
    tld = labels[-1] if labels else ""
    second_level = labels[-2] if len(labels) >= 2 else ""
    # Часть имени хоста без TLD: дефис здесь — сигнал имитации сайтов
    # (TLD исключаем, чтобы punycode-зоны вида xn--p1ai не давали ложный сигнал)
    host_without_tld = ".".join(labels[:-1])

    special_chars = sum(raw.count(ch) for ch in "@?-=&%$!_~;(),*+[]'<>\\|\"")
    digits = sum(ch.isdigit() for ch in raw)
    path_depth = len([seg for seg in path.split("/") if seg])

    return {
        "url_length": len(raw),
        "hostname_length": len(host),
        "has_ip": int(bool(IP_ADDRESS_RE.match(host) or HEX_IP_RE.match(host))),
        "has_at": int("@" in parsed.netloc),
        "has_redirect": int("//" in path),
        "has_hyphen_in_domain": int("-" in host_without_tld),
        "num_dots": raw.count("."),
        "num_subdomains": max(len(labels) - 2, 0),
        "is_https": int(scheme == "https"),
        "num_digits": digits,
        "digit_ratio": digits / len(raw) if raw else 0.0,
        "num_special_chars": special_chars,
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "brand_in_domain": _brand_in_domain(host, second_level),
        "is_shortener": int(host in SHORTENERS),
        "has_punycode": int(host.startswith("xn--") or ".xn--" in host),
        "has_port": int(bool(parsed.port) and parsed.port not in (80, 443)),
        "path_depth": path_depth,
        "query_length": len(query),
        "num_params": len([p for p in query.split("&") if p]) if query else 0,
        "domain_entropy": round(_shannon_entropy(second_level or host), 4),
        "suspicious_words": sum(1 for w in SUSPICIOUS_WORDS if w in raw.lower()),
    }


# ---------------------------------------------------------------------------
# Генерация объяснений (для вердикта на фронтенде)
# ---------------------------------------------------------------------------

def explain_features(url: str, feats: Dict[str, float]) -> List[str]:
    """Возвращает список человекочитаемых объяснений на русском: какие
    признаки URL сработали как подозрительные. Используется бэкендом,
    чтобы пользователь видел не только вердикт, но и его обоснование."""
    explanations: List[str] = []

    if feats["has_ip"]:
        explanations.append("Вместо доменного имени используется IP-адрес — классический приём фишинга")
    if feats["has_at"]:
        explanations.append("Символ «@» в URL: часть до «@» игнорируется браузером, настоящий адрес скрыт")
    if feats["has_redirect"]:
        explanations.append("«//» в пути URL — возможное скрытое перенаправление на другой сайт")
    if feats["has_hyphen_in_domain"]:
        explanations.append("Дефис в доменном имени — часто используется для имитации известных сайтов")
    if feats["suspicious_tld"]:
        explanations.append("Подозрительная доменная зона (TLD), часто встречающаяся у фишинговых сайтов")
    if feats["brand_in_domain"]:
        explanations.append("В домене упоминается известный бренд — возможно, подделка его сайта")
    if feats["is_shortener"]:
        explanations.append("Сокращатель ссылок скрывает настоящий адрес назначения")
    if feats["has_punycode"]:
        explanations.append("Домен записан punycode (xn--) — может имитировать известные домены похожими символами")
    if feats["has_port"]:
        explanations.append("Указан нестандартный сетевой порт")
    if feats["num_subdomains"] >= 3:
        explanations.append(f"Много уровней поддоменов ({feats['num_subdomains']:.0f}) — домен может маскироваться")
    if feats["url_length"] > 100:
        explanations.append(f"Очень длинный URL ({feats['url_length']:.0f} символов) — часто скрывает адрес")
    if feats["num_digits"] > 15:
        explanations.append(f"Много цифр в URL ({feats['num_digits']:.0f}) — признак автогенерированного адреса")
    if feats["domain_entropy"] > 3.2:
        explanations.append(f"Домен выглядит «случайным» (энтропия {feats['domain_entropy']:.2f} бит/символ)")
    if feats["suspicious_words"] >= 2:
        explanations.append(f"Найдено «фишинговых» слов: {feats['suspicious_words']:.0f} (login, verify, account и т.п.)")
    if not feats["is_https"]:
        explanations.append("Соединение без HTTPS — данные передаются открытым текстом")

    return explanations


def feature_table(feats: Dict[str, float]) -> List[dict]:
    """Признаки в виде списка для отображения на фронтенде."""
    return [
        {"name": name, "description": FEATURE_DESCRIPTIONS[name], "value": feats[name]}
        for name in FEATURE_NAMES
    ]
