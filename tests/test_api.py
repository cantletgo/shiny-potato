"""
Тесты HTTP API системы обнаружения фишинга (app/main.py).

Запуск: pytest tests/test_api.py

Тесты используют TestClient (в памяти, без запуска сервера) и требуют
обученную модель (ml/artifacts/model.joblib — появляется после
python ml/train.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestCheckEndpoint:
    def test_phishing_url_detected(self):
        response = client.post(
            "/api/check",
            json={"url": "http://paypal-login.verify-account.tk/update/confirm.php"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "phishing"
        assert data["label"] == 1
        assert data["risk_score"] > 50
        assert 0 <= data["confidence"] <= 1
        assert len(data["explanations"]) > 0

    def test_legit_url_detected(self):
        response = client.post("/api/check", json={"url": "https://github.com"})
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "safe"
        assert data["label"] == 0
        assert data["risk_score"] < 50

    def test_schemeless_url_accepted(self):
        # Пользователь ввёл домен без схемы — система должна понять
        response = client.post("/api/check", json={"url": "yandex.ru"})
        assert response.status_code == 200
        assert response.json()["verdict"] == "safe"

    def test_response_contains_all_features(self):
        response = client.post("/api/check", json={"url": "https://example.com"})
        data = response.json()
        assert len(data["features"]) == 22
        for feature in data["features"]:
            assert set(feature.keys()) == {"name", "description", "value"}

    def test_trusted_top_domain_flagged(self):
        # Домены из топ Majestic помечаются trusted и получают вердикт safe
        response = client.post("/api/check", json={"url": "https://www.google.com"})
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "safe"
        assert data["trusted"] is True
        assert any("Majestic" in item for item in data["explanations"])

    def test_shortener_not_trusted_even_in_top(self):
        # bit.ly — топ-домен Majestic, но сокращатель: скрывает адресата,
        # поэтому доверенным не считается и вердикт отдаёт модель
        response = client.post("/api/check", json={"url": "https://bit.ly/3xK9zLm"})
        assert response.status_code == 200
        data = response.json()
        assert data["trusted"] is False
        assert data["verdict"] == "phishing"

    def test_probabilities_sum_to_one(self):
        data = client.post("/api/check", json={"url": "https://example.com"}).json()
        total = data["probabilities"]["phishing"] + data["probabilities"]["safe"]
        assert abs(total - 1.0) < 0.01

    def test_garbage_without_dot_rejected(self):
        response = client.post("/api/check", json={"url": "абракадабра"})
        assert response.status_code == 400

    def test_too_short_url_rejected(self):
        response = client.post("/api/check", json={"url": "ab"})
        assert response.status_code == 422  # валидация Pydantic (min_length=4)

    def test_missing_field_rejected(self):
        response = client.post("/api/check", json={})
        assert response.status_code == 422


class TestModelInfoEndpoint:
    def test_returns_model_metadata(self):
        response = client.get("/api/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_class"] == "RandomForestClassifier"
        assert "f1" in data["metrics"]
        assert len(data["feature_importances"]) == 22
        assert len(data["features_used"]) == 22


class TestStaticAndHealth:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_index_page_served(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "Веб-система обнаружения фишинговых сайтов" in response.text

    def test_docs_available(self):
        response = client.get("/docs")
        assert response.status_code == 200
