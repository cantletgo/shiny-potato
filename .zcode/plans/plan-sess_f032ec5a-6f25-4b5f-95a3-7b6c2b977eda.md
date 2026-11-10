# Веб-система обнаружения фишинговых сайтов

## Контекст
Учебный проект (вуз/колледж): веб-система, которая по URL определяет, является ли сайт фишинговым. Ядро — ML-модель на признаках URL (RandomForest на scikit-learn), бэкенд — FastAPI, фронтенд — простая одностраничная HTML/JS-страница. Папка `D:\repos\web-phishing-detection-system` пустая, проект создаётся с нуля. Для курсовой важны: объяснимость вердикта («почему сайт признан фишинговым»), метрики качества модели (Accuracy, Precision, Recall, F1, ROC-AUC) и Swagger-документация (в FastAPI идёт бесплатно на `/docs`).

## Структура проекта
```
web-phishing-detection-system/
├── README.md                  # описание, инструкция запуска, скриншот метрик
├── requirements.txt
├── data/
│   ├── raw/                   # скачанные датасеты (в .gitignore)
│   └── processed/             # подготовленный обучающий CSV
├── ml/
│   ├── download_data.py       # сбор датасета ( phishing + легитимные URL)
│   ├── features.py            # извлечение ~18 признаков из URL (переиспользуется и бэкендом!)
│   ├── train.py               # обучение RandomForest, сохранение модели + метрик
│   ├── evaluate.py            # сравнение моделей (RF vs LogReg vs GradBoost), матрица ошибок
│   └── artifacts/             # model.joblib, metrics.json
├── app/
│   ├── main.py                # FastAPI: роуты, раздача статики
│   ├── predictor.py           # загрузка модели, предсказание + объяснения
│   ├── schemas.py             # Pydantic-модели запроса/ответа
│   └── static/
│       ├── index.html         # страница проверки URL
│       ├── style.css
│       └── script.js
└── tests/
    ├── test_features.py
    └── test_api.py
```

## Шаги реализации

### 1. Каркас
`requirements.txt`: fastapi, uvicorn, scikit-learn, pandas, numpy, joblib, pytest, httpx. Виртуальное окружение venv, `pip install`.

### 2. Датасет (`ml/download_data.py`)
- **Фишинговые URL**: OpenPhish community feed (`https://openphish.com/feed.txt`, бесплатный, без регистрации).
- **Легитимные URL**: Majestic Million (`https://downloads.majestic.com/majestic_million.csv`, топ-сайты, бесплатно) — берём домены топ-сайтов.
- Балансировка классов ~1:1, сабсет ~20k–40k строк, сохранение в `data/processed/dataset.csv`.
- Важно: ссылки из фида **не открываются** — анализируется только текст URL (безопасно).
- Если скачивание не сработает — fallback на публичные GitHub-зеркала датасетов Kaggle (CSV вида `url,label`), скрипт поддержит оба формата.

### 3. Признаки (`ml/features.py`)
~18 объяснимых признаков из строки URL (каждый пригодится для защиты курсовой):
длина URL и хоста, IP-адрес вместо домена, символ `@`, `//` в пути, дефис в домене, количество поддоменов/точек, HTTPS, количество цифр и спецсимволов, подозрительный TLD (.tk/.top/.xyz/...), брендовые слова в домене (paypal/apple/bank/sber...), сокращатели ссылок (bit.ly...), punycode `xn--`, порт в URL, глубина пути, длина query, энтропия домена.
Функция `extract_features(url) -> dict` — единая для обучения и для бэкенда (нет расхождения train/serving).

### 4. Обучение (`ml/train.py`, `ml/evaluate.py`)
- Train/test split 80/20, `RandomForestClassifier`.
- `evaluate.py` сравнивает RandomForest / LogisticRegression / GradientBoosting, печатает Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, топ-15 важных признаков (feature_importances_).
- Лучшая модель сохраняется в `ml/artifacts/model.joblib`, метрики в `metrics.json`.

### 5. Бэкенд (`app/`)
- `POST /api/check` — принимает `{"url": "..."}`, извлекает признаки, возвращает: вердикт (phishing/safe), confidence, risk_score, таблицу признаков и **список объяснений на русском** («в URL есть символ @», «домен содержит брендовое слово paypal», «используется подозрительный домен .tk»).
- `GET /api/model/info` — тип модели, метрики, дата обучения (пригодится для отчёта).
- `GET /` — раздача `static/index.html`. Swagger — авто на `/docs`.

### 6. Фронтенд (`app/static/`)
Одна страница на русском: поле ввода URL + кнопка «Проверить», карточка результата (зелёная/красная с confidence-баром), список сработавших признаков, пара пресетов-примеров для быстрых проверок. Чистый HTML/CSS/JS без сборки.

### 7. Тесты (`pytest`)
- `test_features.py`: известные фишинговые и легитимные URL дают ожидаемые признаки.
- `test_api.py`: через TestClient — `/api/check` валидные и невалидные входы.

### 8. README
Описание, схема архитектуры, инструкция запуска (3 команды), метрики модели, примеры запросов.

## Проверка (end-to-end)
1. `python ml/download_data.py && python ml/train.py` — модель обучается, метрики F1 ≥ 0.9 (типично для URL-фич).
2. `uvicorn app.main:app --reload` — сервер поднимается.
3. Проверка в браузере `http://127.0.0.1:8000`: вводим URL из тестового набора — получаем вердикт и объяснения; проверяем `/docs`.
4. `pytest` — все тесты зелёные.
5. Контрольные проверки: `google.com`, `yandex.ru` → safe; URL-паттерны вида `paypal-login.verify-account.tk` → phishing.