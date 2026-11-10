"""
Тесты модуля извлечения признаков (ml/features.py).

Запуск: pytest tests/test_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from features import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    extract_features,
    explain_features,
    feature_table,
    canonical_url,
)


class TestCanonicalUrl:
    def test_adds_scheme(self):
        # Без схемы адрес считается https:// (как современные браузеры)
        assert canonical_url("example.com") == "https://example.com"

    def test_strips_www(self):
        assert canonical_url("https://www.example.com/page") == "https://example.com/page"

    def test_keeps_www_only_domain(self):
        # «www.ru» — настоящий домен, срезать префикс нельзя
        assert canonical_url("http://www.ru").endswith("www.ru")

    def test_lowercase_host(self):
        assert canonical_url("https://EXAMPLE.com/Path") == "https://example.com/Path"


class TestExtractFeatures:
    def test_https_flag(self):
        feats = extract_features("https://example.com")
        assert feats["is_https"] == 1
        assert extract_features("http://example.com")["is_https"] == 0

    def test_ip_address_detected(self):
        feats = extract_features("http://192.168.1.5/bank/login")
        assert feats["has_ip"] == 1

    def test_at_symbol_detected(self):
        feats = extract_features("http://good-site.com@evil.example/login")
        assert feats["has_at"] == 1

    def test_suspicious_tld(self):
        assert extract_features("https://shop-promo.top")["suspicious_tld"] == 1
        assert extract_features("https://example.com")["suspicious_tld"] == 0

    def test_brand_in_domain(self):
        assert extract_features("https://paypal-login.tk")["brand_in_domain"] == 1

    def test_shortener(self):
        assert extract_features("https://bit.ly/3xK9zLm")["is_shortener"] == 1

    def test_punycode(self):
        assert extract_features("https://xn--80ak6aa92e.com")["has_punycode"] == 1

    def test_hyphen_in_domain(self):
        assert extract_features("https://secure-bank.example.com")["has_hyphen_in_domain"] == 1

    def test_url_length_counts(self):
        feats = extract_features("https://example.com/path")
        assert feats["url_length"] == len("https://example.com/path")

    def test_path_depth_and_query(self):
        feats = extract_features("https://example.com/a/b/c?x=1&y=2")
        assert feats["path_depth"] == 3
        assert feats["num_params"] == 2
        assert feats["query_length"] == len("x=1&y=2")

    def test_domain_entropy_positive(self):
        assert extract_features("https://example.com")["domain_entropy"] > 0

    def test_all_features_numeric_and_named(self):
        feats = extract_features("https://example.com")
        assert set(feats.keys()) == set(FEATURE_DESCRIPTIONS.keys())
        assert all(isinstance(v, (int, float)) for v in feats.values())

    def test_schemeless_input(self):
        # Пользователь ввёл домен без схемы — признаки считаются
        feats = extract_features("yandex.ru")
        assert feats["hostname_length"] == len("yandex.ru")


class TestExplainFeatures:
    def test_phishing_url_has_explanations(self):
        url = "http://paypal-login.verify-account.tk/update/confirm.php"
        explanations = explain_features(url, extract_features(url))
        assert len(explanations) >= 3

    def test_clean_url_few_explanations(self):
        url = "https://www.wikipedia.org"
        explanations = explain_features(url, extract_features(url))
        # Чистый https-адрес без тревожных признаков — минимум объяснений
        assert len(explanations) <= 1


class TestFeatureTable:
    def test_table_structure(self):
        feats = extract_features("https://example.com")
        table = feature_table(feats)
        assert len(table) == len(FEATURE_DESCRIPTIONS)
        assert {"name", "description", "value"} == set(table[0].keys())
