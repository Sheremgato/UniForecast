
from datetime import date, timedelta
from pathlib import Path
import calendar
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = ROOT / "data"
MODELS = ROOT / "model"
VERSION = "5.1.0"
MARKER = f"UNIFORECAST_BUILD_{VERSION}"
for asset in ("index.html", "app.js"):
    file = STATIC / asset
    if not file.exists() or MARKER not in file.read_text(encoding="utf-8"):
        raise RuntimeError(f"Версии сайта не совпадают: {file}. Установи весь архив v5 целиком.")

app = FastAPI(title="UniForecast — Calendar MLP API", version=VERSION)

@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

for path in (DATA / "daily_requests.csv", MODELS / "calendar_config.json", MODELS / "calendar_mlp.pt"):
    if not path.exists():
        raise RuntimeError(f"Не хватает файла {path}. Сначала обучи calendar MLP.")

history = pd.read_csv(DATA / "daily_requests.csv", parse_dates=["date"])
history.sort_values(["department", "date"], inplace=True)
if history[["date", "department", "requests"]].isna().any().any() or history.duplicated(["date", "department"]).any():
    raise RuntimeError("Ошибки в data/daily_requests.csv")

with (MODELS / "calendar_config.json").open(encoding="utf-8") as f:
    cfg = json.load(f)
DEPTS = cfg["departments"]
FEATURES = cfg["feature_columns"]
FIDX = {v: i for i, v in enumerate(FEATURES)}
if len(FEATURES) != 26 or sorted(DEPTS) != sorted(history["department"].unique()):
    raise RuntimeError("Несовместимые признаки или подразделения в календарной модели")

class CalendarMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

model = CalendarMLP(len(FEATURES))
model.load_state_dict(torch.load(MODELS / "calendar_mlp.pt", map_location="cpu", weights_only=True))
model.eval()

FIRST = history["date"].min().date()
LAST = history["date"].max().date()
MIN_DATE = date(1900, 1, 1)
MAX_DATE = date(2099, 12, 31)
OBSERVED = {
    name: frame.set_index(frame["date"].dt.date)["requests"].to_dict()
    for name, frame in history.groupby("department")
}

def make_inputs(days, department):
    """Exact same 26 feature columns and formulas as train_calendar_model.py."""
    out = np.zeros((len(days), len(FEATURES)), dtype=np.float32)
    depidx = DEPTS.index(department)
    for i, d in enumerate(days):
        out[i, FIDX[f"month_{d.month}"]] = 1
        out[i, FIDX[f"weekday_{d.weekday()}"]] = 1
        out[i, FIDX[f"department_{depidx}"]] = 1
        ordinal = d.timetuple().tm_yday
        out[i, FIDX["day_sin"]] = np.sin(2 * np.pi * ordinal / 365.25)
        out[i, FIDX["day_cos"]] = np.cos(2 * np.pi * ordinal / 365.25)
    return out


def predict(days, department):
    if not days:
        return np.array([])
    with torch.inference_mode():
        y = model(torch.from_numpy(make_inputs(days, department))).numpy()
    return np.maximum(0.0, y * cfg["target_std"] + cfg["target_mean"])

train = history[history["date"] < "2025-10-01"].copy()
train["month"] = train["date"].dt.month
train["weekend"] = train["date"].dt.weekday >= 5
SEASONAL = train.groupby(["department", "month", "weekend"])["requests"].mean().to_dict()

def baseline(days, department):
    return np.array([SEASONAL[(department, d.month, d.weekday() >= 5)] for d in days], dtype=float)


def scores(actual, p, b):
    mae = float(np.abs(actual-p).mean())
    bmae = float(np.abs(actual-b).mean())
    return {
        "mae": round(mae, 3),
        "rmse": round(float(np.sqrt(np.mean((actual-p)**2))), 3),
        "baseline_mae": round(bmae, 3),
        "baseline_rmse": round(float(np.sqrt(np.mean((actual-b)**2))), 3),
        "improvement_percent": round((bmae-mae)/bmae*100, 2) if bmae else None,
        "n": int(len(actual))
    }

test = history[(history["date"] >= "2026-04-01") & (history["date"] <= "2026-09-30")].copy()
TEST_METRICS = {}
for dept in DEPTS:
    subset = test[test["department"] == dept]
    dates = list(subset["date"].dt.date)
    actual = subset["requests"].to_numpy(dtype=float)
    TEST_METRICS[dept] = scores(actual, predict(dates, dept), baseline(dates, dept))
actual_all, pred_all, base_all = [], [], []
for dept in DEPTS:
    subset = test[test["department"] == dept]
    dates = list(subset["date"].dt.date)
    actual_all.extend(subset["requests"].tolist())
    pred_all.extend(predict(dates, dept).tolist())
    base_all.extend(baseline(dates, dept).tolist())
TEST_TOTAL = scores(np.asarray(actual_all), np.asarray(pred_all), np.asarray(base_all))


def window(kind, chosen):
    if kind == "day":
        return chosen, chosen
    if kind == "week":
        start = chosen - timedelta(days=chosen.weekday())
        return start, start + timedelta(days=6)
    if kind == "month":
        return chosen.replace(day=1), chosen.replace(day=calendar.monthrange(chosen.year, chosen.month)[1])
    raise HTTPException(status_code=422, detail="Период: day, week или month")


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health_status():
    return {"status": "ok", "version": VERSION, "model": "Calendar MLP 26→64→32→1"}


@app.get("/api/departments")
def departments():
    return DEPTS


@app.get("/api/overview")
def overview():
    return {
        "total_requests": int(history["requests"].sum()),
        "total_days": int(history["date"].nunique()),
        "departments": len(DEPTS),
        "data_start": FIRST.isoformat(),
        "data_end": LAST.isoformat(),
        "earliest_predictable_date": MIN_DATE.isoformat(),
        "maximum_supported_date": MAX_DATE.isoformat(),
        "metrics": TEST_TOTAL,
        "model": {
            "features": len(FEATURES),
            "best_epoch": cfg["best_epoch"],
            "validation_mae": round(float(cfg["validation_mae"]), 3),
            "architecture": "26 → 64 → 32 → 1",
        },
        "source": "synthetic",
    }


@app.get("/api/dashboard")
def dashboard(department: str):
    if department not in DEPTS:
        raise HTTPException(status_code=404, detail="Подразделение не найдено")
    frame = history[history["department"] == department].sort_values("date")
    recent = frame.tail(30)
    return {
        "department": department,
        "history": [{"date": row.date.strftime("%Y-%m-%d"), "requests": int(row.requests)} for row in frame.tail(365).itertuples(index=False)],
        "metrics": TEST_METRICS[department],
        "recent_30_total": int(recent["requests"].sum()),
        "recent_30_average": round(float(recent["requests"].mean()), 1),
        "source": "synthetic",
    }


@app.get("/api/period")
def period_forecast(
    department: str,
    period: str = Query(default="week", pattern="^(day|week|month)$"),
    selected_date: date = Query(..., description="Любая дата для дня, недели или месяца"),
):
    if department not in DEPTS:
        raise HTTPException(status_code=404, detail="Подразделение не найдено")
    start, end = window(period, selected_date)
    if start < MIN_DATE or end > MAX_DATE:
        raise HTTPException(status_code=422, detail="Поддерживаются календарные даты с 1900 по 2099 год")
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    raw = predict(days, department)
    rows = []
    observations = OBSERVED[department]
    for d, value in zip(days, raw):
        actual = int(observations[d]) if d in observations else None
        if d < FIRST:
            status = "before_history"
        elif d <= date(2025, 9, 30):
            status = "train"
        elif d <= date(2026, 3, 31):
            status = "validation"
        elif d <= LAST:
            status = "test"
        else:
            status = "future"
        rows.append({
            "date": d.isoformat(),
            "prediction": int(round(float(value))),
            "prediction_exact": round(float(value), 3),
            "actual": actual,
            "mode": "calendar_direct",
            "subset": status,
        })

    actual_rows = [r for r in rows if r["actual"] is not None]
    notes = ["Календарная MLP вычисляет каждый день независимо; рекурсивной цепочки нет."]
    labels = {r["subset"] for r in rows}
    if "train" in labels:
        notes.append("На обучающем периоде прогноз рассчитан по данным, участвовавшим в обучении: это не независимый тест.")
    if "validation" in labels:
        notes.append("Валидационные даты использовались для выбора лучшей эпохи модели.")
    if "future" in labels or "before_history" in labels:
        notes.append("Для дат вне истории нет факта: это сезонный сценарий, а не проверенный прогноз на конкретный далёкий год. Годовой тренд не учитывается.")
    if "test" in labels:
        notes.append("Тестовый период (апрель–сентябрь 2026) не использовался при обучении и выборе эпохи.")
    notes.append(f"На независимом тесте: календарная MLP MAE {TEST_TOTAL['mae']:.3f}, сезонный baseline MAE {TEST_TOTAL['baseline_mae']:.3f}. Меньше — лучше.")
    return {
        "department": department,
        "period": period,
        "selected_date": selected_date.isoformat(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "data_end": LAST.isoformat(),
        "model": "calendar_mlp",
        "rows": rows,
        "summary": {
            "days": len(rows),
            "predicted_total": sum(r["prediction"] for r in rows),
            "daily_average": round(sum(r["prediction"] for r in rows) / len(rows), 2),
            "actual_total": sum(r["actual"] for r in actual_rows) if actual_rows else None,
            "observed_days": len(actual_rows),
            "future_days": len(rows) - len(actual_rows),
            "peak_day": days[int(np.argmax(raw))].isoformat(),
            "mae_known_days": round(float(np.mean([abs(r["prediction_exact"] - r["actual"]) for r in actual_rows])), 3) if actual_rows else None,
        },
        "notes": notes,
        "source": "synthetic",
    }
