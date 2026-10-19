// UNIFORECAST_BUILD_5.1.0 — browser/API must be same release.
"use strict";

const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const fmt = new Intl.NumberFormat("ru-RU");
const weekday = ["вс", "пн", "вт", "ср", "чт", "пт", "сб"];
const weekdayLong = ["Воскресенье", "Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота"];
let overview = null;
let current = null;
let activeView = "dashboard";
let periodParams = { period: "week", selected_date: null };
let requestId = 0;
let currentPeriod = null;
const pad2 = n => String(n).padStart(2, "0");
const localISO = d => [d.getFullYear(), pad2(d.getMonth() + 1), pad2(d.getDate())].join("-");
const parseISO = s => new Date(`${s}T12:00:00`);
const dateLong = s => dateText(s, { day: "numeric", month: "long", year: "numeric" });

function dateText(iso, options = { day: "2-digit", month: "short" }) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString("ru-RU", options);
}
function numeric(value, fractionDigits = 0) {
  return typeof value === "number" ? value.toLocaleString("ru-RU", { minimumFractionDigits: fractionDigits, maximumFractionDigits: fractionDigits }) : "—";
}
function setText(id, value) { $(id).textContent = value; }
async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) {
    let detail = `Ошибка API ${response.status}`;
    try { const data = await response.json(); if (typeof data.detail === "string") detail = data.detail; } catch (_) {}
    throw new Error(detail);
  }
  return await response.json();
}
function showError(message) { const banner = $("errorBanner"); banner.hidden = false; banner.textContent = message; }
function clearError() { $("errorBanner").hidden = true; }

function svgEl(tag, attrs = {}, text) {
  const el = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  if (text != null) el.textContent = text;
  return el;
}
function drawHistory(items) {
  const svg = $("historyChart");
  svg.replaceChildren();
  const tooltip = $("chartTooltip");
  tooltip.hidden = true;
  if (!Array.isArray(items) || items.length === 0) return;

  // A full year of daily observations, with a readable 7-day moving average.
  const width = 900, height = 305;
  const pad = { left: 46, top: 20, right: 18, bottom: 43 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  const values = items.map(d => d.requests);
  const smoothed = values.map((_, i) => {
    const from = Math.max(0, i - 6);
    const week = values.slice(from, i + 1);
    return week.reduce((a, b) => a + b, 0) / week.length;
  });
  const maxY = Math.max(5, ...values, ...smoothed);
  const upper = Math.ceil(maxY / 10) * 10;
  const x = i => pad.left + i * w / Math.max(1, items.length - 1);
  const y = n => pad.top + h - n / upper * h;

  for (let k = 0; k <= 4; k++) {
    const v = upper * k / 4;
    const yy = y(v);
    svg.appendChild(svgEl("line", {
      x1: pad.left, x2: width - pad.right, y1: yy, y2: yy,
      stroke: "#e7edf5", "stroke-dasharray": "4 5"
    }));
    svg.appendChild(svgEl("text", {
      x: pad.left - 11, y: yy + 4, "text-anchor": "end",
      fill: "#9aaac0", "font-size": 11
    }, numeric(v)));
  }

  // Monthly labels (rather than 365 crowded day labels).
  const ticks = items.map((d, i) => ({ d, i })).filter(({ d, i }) => i === 0 || d.date.endsWith("-01"));
  for (const { d, i } of ticks) {
    const xx = x(i);
    svg.appendChild(svgEl("line", { x1: xx, x2: xx, y1: pad.top, y2: pad.top + h, stroke: "#f1f4f9" }));
    svg.appendChild(svgEl("text", {
      x: xx, y: height - 14, "text-anchor": i === 0 ? "start" : "middle",
      fill: "#94a3b8", "font-size": 10
    }, dateText(d.date, { month: "short" })));
  }

  const line = series => series.map((n, i) => `${i ? "L" : "M"}${x(i).toFixed(2)} ${y(n).toFixed(2)}`).join(" ");
  const smoothPath = line(smoothed);
  const rawPath = line(values);
  const defs = svgEl("defs");
  const gradient = svgEl("linearGradient", { id: "historyFill", x1: "0%", y1: "0%", x2: "0%", y2: "100%" });
  gradient.appendChild(svgEl("stop", { offset: "0%", "stop-color": "#5384f3", "stop-opacity": "0.17" }));
  gradient.appendChild(svgEl("stop", { offset: "100%", "stop-color": "#5384f3", "stop-opacity": "0" }));
  defs.appendChild(gradient);
  svg.appendChild(defs);
  svg.appendChild(svgEl("path", {
    d: `${smoothPath} L ${x(items.length - 1)} ${y(0)} L ${x(0)} ${y(0)} Z`,
    fill: "url(#historyFill)"
  }));
  svg.appendChild(svgEl("path", { d: rawPath, fill: "none", stroke: "#bac9e3", "stroke-width": 1, opacity: .75 }));
  svg.appendChild(svgEl("path", { d: smoothPath, fill: "none", stroke: "#366bea", "stroke-width": 2.8, "stroke-linejoin": "round", "stroke-linecap": "round" }));

  // Single hit surface avoids 365 overlapping hover targets.
  const guide = svgEl("line", { y1: pad.top, y2: pad.top + h, stroke: "#8ea6d5", "stroke-dasharray": "4 4", visibility: "hidden" });
  const dot = svgEl("circle", { r: 5, fill: "#366bea", stroke: "white", "stroke-width": 2, visibility: "hidden" });
  const hit = svgEl("rect", { x: pad.left, y: pad.top, width: w, height: h, fill: "transparent", tabindex: "0", role: "img", "aria-label": "Годовой график обращений. Используйте стрелки влево и вправо для просмотра данных по дням." });
  svg.append(guide, dot, hit);
  let active = items.length - 1;
  function focusDay(i) {
    active = Math.max(0, Math.min(items.length - 1, i));
    const item = items[active];
    guide.setAttribute("x1", x(active)); guide.setAttribute("x2", x(active));
    guide.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(active)); dot.setAttribute("cy", y(smoothed[active]));
    dot.setAttribute("visibility", "visible");
    tooltip.textContent = `${dateText(item.date, { day: "numeric", month: "long", year: "numeric" })} · ${item.requests} за день · среднее за 7 дней: ${numeric(smoothed[active], 1)}`;
    tooltip.style.left = `${Math.min(77, Math.max(10, x(active) / width * 100))}%`;
    tooltip.style.top = `${Math.max(2, y(smoothed[active]) / height * 100 - 17)}%`;
    tooltip.hidden = false;
  }
  const hide = () => { guide.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tooltip.hidden = true; };
  hit.addEventListener("pointermove", event => {
    const rect = svg.getBoundingClientRect();
    const svgX = (event.clientX - rect.left) / rect.width * width;
    const i = Math.round((svgX - pad.left) / w * (items.length - 1));
    focusDay(i);
  });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => focusDay(active));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", e => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      focusDay(active + (e.key === "ArrowRight" ? 1 : -1));
    }
  });
}
function barNode(row, detailed, max) {
  const isWeekend = [0, 6].includes(new Date(`${row.date}T12:00:00`).getDay());
  const wrapper = document.createElement("div");
  wrapper.className = `${detailed ? "forecast-detail-item" : "forecast-item"}${isWeekend ? " weekend" : ""}`;
  const date = document.createElement("span"); date.className = "forecast-day"; date.textContent = dateText(row.date);
  const track = document.createElement("div"); track.className = "forecast-track";
  const bar = document.createElement("div"); bar.className = "forecast-fill"; bar.style.width = `${Math.max(2, row.prediction / Math.max(1, max) * 100)}%`;
  track.appendChild(bar);
  const value = document.createElement(detailed ? "b" : "span");
  value.className = "forecast-count"; value.textContent = fmt.format(row.prediction);
  wrapper.append(date, track, value); return wrapper;
}
function showDashboard(data) {
  if (!data || !Array.isArray(data.history) || !data.metrics) {
    throw new Error("API dashboard несовместим с сайтом версии 5.1. Переустанови файлы из одного архива.");
  }
  setText("recentTotal", fmt.format(data.recent_30_total));
  setText("recentAvg", numeric(data.recent_30_average, 1));
  setText("departmentMae", numeric(data.metrics.mae, 3));
  setText("departmentGain", data.metrics.improvement_percent == null ? "—" : `${numeric(data.metrics.improvement_percent, 2)}%`);
  drawHistory(data.history);
}
function showPeriod(data) {
  currentPeriod = data;
  if (!data || !Array.isArray(data.rows) || !data.summary) {
    throw new Error("API вернул несовместимый формат периода. Проверь версию 5.0 и полную установку файлов.");
  }
  const rows = data.rows;
  if (!rows.length) throw new Error("Не удалось получить данные за выбранный период.");
  const s = data.summary;
  const max = Math.max(...rows.map(r => r.prediction), 1);
  const preview = rows.slice(0, Math.min(7, rows.length));
  $("forecastBars").replaceChildren(...preview.map(row => barNode(row, false, max)));
  setText("forecastShortTitle", data.period === "day" ? "Выбранный день" : data.period === "week" ? "Выбранная неделя" : "Выбранный месяц");
  setText("forecastInterval", `${dateText(data.start_date)} – ${dateLong(data.end_date)} · показаны первые ${preview.length} дн.`);
  $("forecastDetailBars").replaceChildren(...rows.map(row => barNode(row, true, max)));
  const table = $("forecastTable");
  table.replaceChildren();
  rows.forEach(row => {
    const tr = document.createElement("tr");
    if (row.actual === null) tr.className = "future-row";
    const weekDay = new Date(`${row.date}T12:00:00`).getDay();
    const values = [dateLong(row.date), weekdayLong[weekDay], fmt.format(row.prediction), row.actual === null ? "—" : fmt.format(row.actual)];
    values.forEach(value => { const td = document.createElement("td"); td.textContent = value; tr.appendChild(td); });
    table.appendChild(tr);
  });
  const range = data.start_date === data.end_date
    ? dateLong(data.start_date) : `${dateLong(data.start_date)} — ${dateLong(data.end_date)}`;
  setText("forecastWindow", range);
  setText("periodPredicted", fmt.format(s.predicted_total));
  setText("periodAverage", numeric(s.daily_average, 1));
  setText("periodActual", s.actual_total == null ? "—" : fmt.format(s.actual_total));
  setText("periodActualHint", s.observed_days === 0 ? "Нет наблюдений в будущие даты" : `Факт за ${s.observed_days} из ${s.days} дней`);
  setText("periodPeak", s.peak_day ? dateText(s.peak_day) : "—");
}
function showOverview(data) {
  setText("historyEnd", dateText(data.data_end, { day: "2-digit", month: "long", year: "numeric" }));
  setText("bestEpoch", numeric(data.model.best_epoch));
  setText("validationMae", numeric(data.model.validation_mae, 3));
  setText("featuresCount", numeric(data.model.features));
  setText("overallMae", numeric(data.metrics.mae, 3));
  setText("overallBaseline", numeric(data.metrics.baseline_mae, 3));
  setText("overallRmse", numeric(data.metrics.rmse, 3));
  setText("overallGain", data.metrics.improvement_percent == null ? "—" : `${numeric(data.metrics.improvement_percent, 2)}%`);
}
async function selectDepartment(name) {
  const id = ++requestId;
  clearError();
  try {
    const params = new URLSearchParams({ department: name, period: periodParams.period, selected_date: periodParams.selected_date });
    const [dashboard, period] = await Promise.all([
      getJSON(`/api/dashboard?department=${encodeURIComponent(name)}`),
      getJSON(`/api/period?${params}`)
    ]);
    if (id !== requestId) return;
    current = dashboard;
    showDashboard(dashboard);
    showPeriod(period);
  } catch (error) {
    if (id === requestId) showError(`Не удалось рассчитать период: ${error.message}`);
  }
}
function changeView(view) {
  activeView = view;
  const titles = {
    dashboard: ["Обзор нагрузки", "Мониторинг обращений и прогноз с помощью нейронной сети."],
    forecast: ["Прогнозирование", "Выберите любой календарный день, неделю или месяц."],
    model: ["Модель и качество", "Архитектура MLP, методология обучения и результаты тестирования."]
  };
  for (const name of Object.keys(titles)) $("view-" + name).classList.toggle("active", name === view);
  document.querySelectorAll(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  setText("pageTitle", titles[view][0]); setText("breadcrumb", titles[view][0]); setText("pageSubtitle", titles[view][1]);
}
function adjustInputVisibility() {
  const isMonth = periodParams.period === "month";
  $("dateInputGroup").hidden = isMonth;
  $("monthInputGroup").hidden = !isMonth;
  setText("dateInputLabel", periodParams.period === "week" ? "Дата внутри недели" : "Выберите день");
  document.querySelectorAll(".period-mode").forEach(button => {
    const active = button.dataset.period === periodParams.period;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $("startDate").value = periodParams.selected_date;
  $("monthDate").value = periodParams.selected_date.slice(0, 7);
}
function readPeriodInput() {
  const value = periodParams.period === "month" ? $("monthDate").value + "-01" : $("startDate").value;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  periodParams.selected_date = value;
  return true;
}
function movePeriod(delta) {
  if (!readPeriodInput()) return;
  const dt = parseISO(periodParams.selected_date);
  if (periodParams.period === "month") {
    dt.setDate(1);
    dt.setMonth(dt.getMonth() + delta);
  } else {
    dt.setDate(dt.getDate() + delta * (periodParams.period === "week" ? 7 : 1));
  }
  periodParams.selected_date = localISO(dt);
  adjustInputVisibility();
  selectDepartment($("departmentSelect").value);
}
async function init() {
  document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => changeView(button.dataset.view)));
  $("openForecast").addEventListener("click", () => changeView("forecast"));
  $("departmentSelect").addEventListener("change", event => selectDepartment(event.target.value));
  document.querySelectorAll(".period-mode").forEach(button => button.addEventListener("click", () => {
    if (!readPeriodInput()) return;
    periodParams.period = button.dataset.period;
    adjustInputVisibility();
    selectDepartment($("departmentSelect").value);
  }));
  $("previousPeriod").addEventListener("click", () => movePeriod(-1));
  $("nextPeriod").addEventListener("click", () => movePeriod(1));
  $("forecastForm").addEventListener("submit", async event => {
    event.preventDefault();
    if (!readPeriodInput()) return;
    const button = $("forecastRun");
    button.disabled = true;
    button.textContent = "Рассчитываем…";
    try { await selectDepartment($("departmentSelect").value); }
    finally { button.disabled = false; button.textContent = "Показать прогноз →"; }
  });
  try {
    const [health, departments, initialOverview] = await Promise.all([
      getJSON("/api/health"), getJSON("/api/departments"), getJSON("/api/overview")
    ]);
    if (health.version !== "5.1.0") {
      throw new Error(`Сайт версии 5.1, а API версии ${health.version || "не определена"}. Установи все файлы из одного архива.`);
    }
    if (!Array.isArray(departments) || departments.length === 0) {
      throw new Error("API /api/departments должен вернуть список подразделений.");
    }
    overview = initialOverview;
    // Current calendar week by default, but no fixed start/end or 7-day-only limit.
    periodParams.selected_date = localISO(new Date());
    $("startDate").min = overview.earliest_predictable_date;
    $("startDate").max = overview.maximum_supported_date;
    $("monthDate").max = overview.maximum_supported_date.slice(0, 7);
    adjustInputVisibility();
    setText("knownThrough", dateLong(overview.data_end));
    setText("serverStatus", health.status === "ok" ? "активен" : "недоступен");
    const select = $("departmentSelect");
    select.replaceChildren(...departments.map(name => {
      const option = document.createElement("option"); option.value = name; option.textContent = name; return option;
    }));
    const initialDept = departments.includes("Учебный отдел") ? "Учебный отдел" : departments[0];
    select.value = initialDept;
    showOverview(overview);
    await selectDepartment(initialDept);
  } catch (error) {
    setText("serverStatus", "ошибка");
    showError(`Ошибка подключения к API: ${error.message}. Проверь запуск сервера и файлы проекта.`);
  }
}
document.addEventListener("DOMContentLoaded", init);
