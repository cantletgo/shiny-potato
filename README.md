# Веб-система обнаружения фишинговых сайтов

Учебный проект: веб-приложение, которое по одному URL определяет, является ли
сайт фишинговым, и **объясняет вердикт** — какие признаки адреса сработали.

## Как это работает

```
        URL пользователя
               │
               ▼
┌──────────────────────────────┐
│  ml/features.py              │   22 признака URL: длина, IP вместо домена,
│  извлечение признаков        │   «@», дефисы, TLD, бренды, цифры, энтропия…
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  ml/artifacts/model.joblib   │     │  известные домены (топ-1000  │
│  RandomForest (sklearn)      │     │  Majestic) — trusted-сигнал  │
└──────────────┬───────────────┘     └──────────────┬───────────────┘
               │                                    │
               ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  app/main.py (FastAPI): вердикт + уверенность + объяснения      │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
        Веб-интерфейс (HTML/JS) и REST API (/docs — Swagger)
```

Модель: **RandomForestClassifier** (scikit-learn), обучена на 40 000 URL
(20 000 фишинговых + 20 000 легитимных).

**Метрики на тестовой выборке (20 %):**

| Метрика   | Значение |
|-----------|----------|
| Accuracy  | 97.46 %  |
| Precision | 98.82 %  |
| Recall    | 96.07 %  |
| F1-мера   | 97.43 %  |
| ROC-AUC   | 99.35 %  |

Сравнение моделей (RandomForest / LogisticRegression / GradientBoosting) —
`ml/evaluate.py`, результаты в `ml/artifacts/comparison.json`.

## Структура проекта

```
├── ml/                        # ML-ядро
│   ├── download_data.py       # сбор датасета (UCI PhiUSIIL + OpenPhish + Majestic)
│   ├── features.py            # извлечение 22 признаков + канонизация URL
│   ├── train.py               # обучение RandomForest, сохранение артефактов
│   ├── evaluate.py            # сравнение 3 моделей, confusion matrix
│   ├── dataset.py             # загрузка датасета с кэшем признаков
│   └── artifacts/             # model.joblib, metrics.json, known_good_domains.txt
├── app/                       # бэкенд FastAPI
│   ├── main.py                # POST /api/check, GET /api/model/info, /health
│   ├── predictor.py           # предсказание + белый список доменов
│   ├── schemas.py             # Pydantic-схемы API
│   └── static/                # фронтенд (HTML/CSS/JS, без сборки)
├── tests/                     # pytest: признаки и API (34 теста)
└── data/                      # сырые данные и обучающий датасет
```

## Запуск

Требуется Python 3.10+.

```bash
# 1. Виртуальное окружение и зависимости
python -m venv venv
venv\Scripts\activate            # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

# 2. Датасет (UCI PhiUSIIL ~15 МБ + OpenPhish + Majestic Million ~80 МБ)
python ml/download_data.py

# 3. Обучение модели
python ml/train.py

# 4. Запуск веб-системы
uvicorn app.main:app --reload
```

Интерфейс: **http://127.0.0.1:8000** · Swagger: **http://127.0.0.1:8000/docs**

Тесты: `python -m pytest` (34 теста: признаки и API).

## Примеры запросов к API

```bash
# Проверка URL
curl -X POST http://127.0.0.1:8000/api/check \
     -H "Content-Type: application/json" \
     -d '{"url": "http://paypal-login.verify-account.tk/update/confirm.php"}'
```

Ответ:

```json
{
  "url": "http://paypal-login.verify-account.tk/update/confirm.php",
  "verdict": "phishing",
  "label": 1,
  "trusted": false,
  "confidence": 0.9984,
  "risk_score": 99.8,
  "probabilities": { "phishing": 0.9984, "safe": 0.0016 },
  "explanations": [
    "Дефис в доменном имени — часто используется для имитации известных сайтов",
    "Подозрительная доменная зона (TLD), часто встречающаяся у фишинговых сайтов",
    "В домене упоминается известный бренд — возможно, подделка его сайта"
  ]
}
```

`GET /api/model/info` — метрики, дата обучения, важность признаков.

## Признаки URL (22)

Числовые характеристики адреса, вычисляются одинаково при обучении и
обслуживании (`ml/features.py`): длина URL и хоста, IP-адрес вместо домена,
символ «@», «//» в пути, дефис в домене, количество точек/поддоменов/цифр/спецсимволов,
HTTPS, подозрительный TLD (.tk, .top, .xyz…), брендовое слово в чужом домене,
сокращатель ссылок, punycode (xn--), нестандартный порт, глубина пути, длина
и число параметров запроса, энтропия домена, «фишинговые» слова (login, verify…).

## Как боролись с ложными срабатываниями

Итерации по данным — главная часть работы над качеством:

1. **«www.»-перекос**: в PhiUSIIL легитимные адреса записаны с «www.», без путей
   и query — модель учила «нет www → фишинг» и блокировала `yandex.ru`, `vk.com`.
   Решение: канонизация (срезка «www.», нижний регистр хоста) + аугментация
   легитимных URL путями страниц.
2. **«query → фишинг»**: в легитимном классе не было ни одного параметра запроса
   (`google.com/search?q=…` блокировался). Добавлены реалистичные query-варианты.
3. **«hex-хвост → фишинг»**: `chatgpt.com/c/<uuid>` и `notion.so/…-<hex>`
   блокировались из-за цифр в пути. Добавлены варианты с UUID/hex-путями.
4. **Бренды в подстроках**: признак «бренд в домене» ловил и официальный домен
   (`sberbank.ru`). Теперь — точное совпадение сегментов: бренд-домен ≠ сигнал,
   `brand-в-чужом-домене` (paypal-login.tk) — сигнал.
5. **Доверенные домены**: топ-1000 сайтов Majestic Million помечаются
   `trusted=true` независимо от модели (кроме сокращателей ссылок — bit.ly
   скрывает адресата и проверяется моделью).

Актуальный список проверок — в `tests/test_features.py` и `tests/test_api.py`.

## Ограничения и развитие

- Модель анализирует только строку URL, не содержимое страницы и не возраст
  домена (WHIOS). Датасет PhiUSIIL содержит перекос «https у легитимных»,
  поэтому `is_https` — самый важный признак (0.39); в реальности фишинг всё
  чаще за TLS-сертификатами, и признак следует дополнять проверкой сертификата.
- Короткие «новые» домены, не попавшие в топ Majestic, могут потребовать
  дообучения на свежем фиде.
- Направления развития: проверка содержимого страницы (HTML-признаки), API
  Google Safe Browsing как дополнительный сигнал, ансамбль с BERT-моделью
  (например, `ealvaradob/bert-finetuned-phishing` с Hugging Face), браузерное
  расширение, история проверок и дашборд.

## Источники данных

- [PhiUSIIL Phishing URL Dataset (UCI)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) — 235 795 URL
- [OpenPhish community feed](https://openphish.com/) — актуальные фишинговые ссылки
- [Majestic Million](https://majestic.com/reports/majestic-million) — топ сайтов для легитимного класса и доверенного списка
- Резерв: [репозиторий shreyagopal](https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques)
