
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODEL = BASE / "model"
CHARTS = BASE / "charts"
CHARTS.mkdir(exist_ok=True)

with open(MODEL / "calendar_config.json", encoding="utf-8") as f:
    config = json.load(f)

df = pd.read_csv(DATA / "daily_requests.csv", parse_dates=["date"])
if df[["date", "department", "requests"]].isna().any().any():
    raise ValueError("Missing essential data")
if df.duplicated(["date", "department"]).any():
    raise ValueError("Duplicate rows for date/department")

train = df[df["date"] < "2025-10-01"].copy()
test = df[(df["date"] >= "2026-04-01") & (df["date"] <= "2026-09-30")].copy()
if train.empty or test.empty:
    raise ValueError("Train or test data missing")

departments = config["departments"]
feature_columns = config["feature_columns"]

def make_features(frame):
    dates = pd.to_datetime(frame["date"])
    features = pd.DataFrame(index=frame.index)
    for month in range(1, 13):
        features[f"month_{month}"] = (dates.dt.month == month).astype("float32")
    for weekday in range(7):
        features[f"weekday_{weekday}"] = (dates.dt.dayofweek == weekday).astype("float32")
    for i, name in enumerate(departments):
        features[f"department_{i}"] = (frame["department"] == name).astype("float32")
    day_of_year = dates.dt.dayofyear.to_numpy(dtype="float32")
    features["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return features[feature_columns]

class CalendarMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

model = CalendarMLP(len(feature_columns))
model.load_state_dict(torch.load(MODEL / "calendar_mlp.pt", map_location="cpu", weights_only=True))
model.eval()
with torch.inference_mode():
    x = torch.from_numpy(make_features(test).to_numpy(dtype="float32"))
    predictions = model(x).numpy() * config["target_std"] + config["target_mean"]
predictions = np.maximum(predictions, 0.0)

train = train.assign(month=train["date"].dt.month, weekend=train["date"].dt.dayofweek >= 5)
test = test.assign(month=test["date"].dt.month, weekend=test["date"].dt.dayofweek >= 5)
means = train.groupby(["department", "month", "weekend"])["requests"].mean()

baseline = []
for row in test.itertuples():
    key = (row.department, row.month, row.weekend)
    if key not in means.index:
        raise ValueError(f"Insufficient training data for seasonal baseline: {key}")
    baseline.append(float(means.loc[key]))
baseline = np.asarray(baseline, dtype=float)

result = test[["date", "department", "requests"]].copy()
result["calendar_prediction"] = predictions
result["seasonal_baseline"] = baseline
result.to_csv(DATA / "calendar_test_predictions.csv", index=False)


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred))), float(np.sqrt(np.mean((y - pred) ** 2)))

print("=== CALENDAR MODEL: INDEPENDENT TEST ===")
print(f"Dates: {test['date'].min().date()} .. {test['date'].max().date()}")
print(f"Observations: {len(test)}")
y = result["requests"].to_numpy()
mlp_mae, mlp_rmse = metrics(y, result["calendar_prediction"])
base_mae, base_rmse = metrics(y, result["seasonal_baseline"])
print(f"Calendar MLP MAE:       {mlp_mae:.3f}")
print(f"Calendar MLP RMSE:      {mlp_rmse:.3f}")
print(f"Seasonal baseline MAE:  {base_mae:.3f}")
print(f"Seasonal baseline RMSE: {base_rmse:.3f}")
if base_mae > 0:
    print(f"MAE improvement:         {(base_mae - mlp_mae) / base_mae * 100:+.2f}%")

print("\n=== BY DEPARTMENT ===")
for dept, rows in result.groupby("department"):
    a, _ = metrics(rows["requests"], rows["calendar_prediction"])
    b, _ = metrics(rows["requests"], rows["seasonal_baseline"])
    print(f"{dept}: MLP MAE={a:.3f}, baseline MAE={b:.3f}")

july = result[(result["date"].dt.year == 2026) & (result["date"].dt.month == 7)
              & (result["department"] == "Приемная комиссия")].sort_values("date")
if july.empty:
    print("\nJuly 2026 admissions test segment not available")
else:
    print("\n=== JULY 2026 ADMISSIONS ===")
    print(f"Observed mean:  {july['requests'].mean():.2f} per day")
    print(f"Calendar MLP:   {july['calendar_prediction'].mean():.2f} per day")
    print(f"Baseline:       {july['seasonal_baseline'].mean():.2f} per day")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(july["date"], july["requests"], label="Actual")
    ax.plot(july["date"], july["calendar_prediction"], label="Calendar MLP")
    ax.plot(july["date"], july["seasonal_baseline"], label="Seasonal baseline", linestyle="--")
    ax.set_title("Admissions: July 2026 (held-out test)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily requests")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS / "calendar_test_july.png", dpi=150)
    plt.close(fig)

print("\nSaved: data/calendar_test_predictions.csv")
print("Saved: charts/calendar_test_july.png (if July segment available)")
print("No model retraining and no old files overwritten.")
