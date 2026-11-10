/* Логика веб-интерфейса системы обнаружения фишинга.
   Общается с бэкендом только через POST /api/check и GET /api/model/info. */

"use strict";

const form = document.getElementById("check-form");
const input = document.getElementById("url-input");
const button = document.getElementById("check-btn");
const resultBlock = document.getElementById("result");
const verdictCard = document.getElementById("verdict-card");
const verdictBadge = document.getElementById("verdict-badge");
const resultUrl = document.getElementById("result-url");
const resultModel = document.getElementById("result-model");
const riskValue = document.getElementById("risk-value");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceCaption = document.getElementById("confidence-caption");
const explanationsBlock = document.getElementById("explanations-block");
const explanationsList = document.getElementById("explanations-list");
const noWarnings = document.getElementById("no-warnings");
const featuresTbody = document.getElementById("features-tbody");
const aboutSection = document.getElementById("about");
const aboutGrid = document.getElementById("about-grid");
const metricsTbody = document.getElementById("metrics-tbody");
const importancesBlock = document.getElementById("importances");

const METRIC_LABELS = {
  accuracy: "Accuracy (доля верных ответов)",
  precision: "Precision (точность тревог)",
  recall: "Recall (полнота поиска фишинга)",
  f1: "F1-мера",
  roc_auc: "ROC-AUC",
};

let loadingNote = null;

/* ---------- Проверка URL ---------- */

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = input.value.trim();
  if (!url) return;

  setLoading(true);
  try {
    const response = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.detail || "Сервис вернул ошибку");
      return;
    }
    renderResult(data);
    loadModelInfoOnce();
  } catch (err) {
    showError("Не удалось связаться с сервером: " + err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(state) {
  button.disabled = state;
  if (state) {
    // Новая проверка: убираем прошлое сообщение об ошибке и показываем индикатор
    document.querySelectorAll(".error-note").forEach((el) => el.remove());
    loadingNote = document.createElement("div");
    loadingNote.className = "loading-note";
    loadingNote.textContent = "Анализирую URL…";
    resultBlock.before(loadingNote);
  } else if (loadingNote) {
    // Убираем только индикатор загрузки — сообщение об ошибке должно остаться
    loadingNote.remove();
    loadingNote = null;
  }
}

function showError(message) {
  document.querySelectorAll(".loading-note, .error-note").forEach((el) => el.remove());
  resultBlock.classList.add("hidden");
  const note = document.createElement("div");
  note.className = "error-note";
  note.textContent = "⚠ " + message;
  resultBlock.before(note);
}

function fmt(value) {
  if (Number.isInteger(Number(value))) return String(value);
  return String(Number(value).toFixed(3));
}

function renderResult(data) {
  const isPhishing = data.verdict === "phishing";

  verdictCard.className = "verdict-card " + data.verdict;
  verdictBadge.className = "verdict-badge " + data.verdict;
  verdictBadge.textContent = isPhishing
    ? "⚠ ФИШИНГ"
    : (data.trusted ? "✓ ИЗВЕСТНЫЙ САЙТ" : "✓ БЕЗОПАСНО");
  resultUrl.textContent = data.url;
  resultModel.textContent =
    "Модель: " + data.model.class +
    (data.model.f1 ? " · F1 = " + data.model.f1 : "");

  riskValue.textContent = data.risk_score.toFixed(1) + "%";
  riskValue.className = "risk-value " + (isPhishing ? "high" : "low");

  confidenceFill.className = "confidence-fill " + data.verdict;
  confidenceFill.style.width = (data.confidence * 100).toFixed(1) + "%";
  confidenceCaption.textContent =
    "Уверенность модели: " + (data.confidence * 100).toFixed(1) + "% · " +
    "вероятность фишинга: " + (data.probabilities.phishing * 100).toFixed(1) + "%";

  explanationsList.innerHTML = "";
  if (data.explanations.length > 0) {
    for (const item of data.explanations) {
      const li = document.createElement("li");
      li.textContent = item;
      explanationsList.appendChild(li);
    }
    noWarnings.classList.add("hidden");
  } else {
    noWarnings.classList.remove("hidden");
  }
  explanationsBlock.classList.remove("hidden");

  featuresTbody.innerHTML = "";
  for (const f of data.features) {
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td");
    nameTd.textContent = f.name;
    const descTd = document.createElement("td");
    descTd.textContent = f.description;
    const valTd = document.createElement("td");
    valTd.className = "value";
    valTd.textContent = fmt(f.value);
    tr.append(nameTd, descTd, valTd);
    featuresTbody.appendChild(tr);
  }

  document.querySelectorAll(".loading-note, .error-note").forEach((el) => el.remove());
  resultBlock.classList.remove("hidden");
  resultBlock.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- Кнопки-примеры ---------- */

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.url;
    form.dispatchEvent(new Event("submit", { cancelable: true }));
  });
});

/* ---------- Информация о модели ---------- */

let modelInfoLoaded = false;

async function loadModelInfoOnce() {
  if (modelInfoLoaded) return;
  modelInfoLoaded = true;
  try {
    const response = await fetch("/api/model/info");
    const info = await response.json();
    renderModelInfo(info);
  } catch (err) {
    /* блок «О модели» не критичен для работы страницы */
  }
}

document.getElementById("about-link").addEventListener("click", () => {
  loadModelInfoOnce();
});

function renderModelInfo(info) {
  const ds = info.dataset || {};
  aboutGrid.innerHTML = "";

  const items = [
    ["Класс модели", info.model_class],
    [ds.n_train ? "Обучающих примеров" : "—", ds.n_train || "—"],
    [ds.n_test ? "Тестовых примеров" : "—", ds.n_test || "—"],
    [info.trained_at_utc ? "Дата обучения (UTC)" : "—",
     info.trained_at_utc ? new Date(info.trained_at_utc).toLocaleString("ru-RU") : "—"],
  ];
  for (const [label, value] of items) {
    const div = document.createElement("div");
    div.className = "about-item";
    div.innerHTML =
      '<div class="num">' + value + "</div><div class=\"label\">" + label + "</div>";
    aboutGrid.appendChild(div);
  }

  metricsTbody.innerHTML = "";
  for (const [key, label] of Object.entries(METRIC_LABELS)) {
    if (!(key in (info.metrics || {}))) continue;
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    td1.textContent = label;
    const td2 = document.createElement("td");
    td2.textContent = (info.metrics[key] * 100).toFixed(2) + "%";
    tr.append(td1, td2);
    metricsTbody.appendChild(tr);
  }

  importancesBlock.innerHTML = "";
  const tops = (info.feature_importances || []).slice(0, 10);
  const maxImp = tops.length ? tops[0].importance : 1;
  for (const item of tops) {
    const row = document.createElement("div");
    row.className = "importance-row";
    row.innerHTML =
      '<span class="importance-name">' + item.name + "</span>" +
      '<span class="importance-bar-track"><span class="importance-bar-fill" style="width:' +
      ((item.importance / maxImp) * 100).toFixed(1) + '%"></span></span>' +
      '<span class="importance-val">' + item.importance.toFixed(3) + "</span>";
    importancesBlock.appendChild(row);
  }

  aboutSection.classList.remove("hidden");
}
