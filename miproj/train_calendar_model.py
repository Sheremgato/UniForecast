from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "daily_requests.csv"
OUT = BASE / "model"
OUT.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

df = pd.read_csv(DATA, parse_dates=["date"])
required = {"date", "department", "requests"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")
if df[list(required)].isna().any().any():
    raise ValueError("Missing date, department, or request count")
if df.duplicated(["date", "department"]).any():
    raise ValueError("Duplicate (date, department) pairs")
if (df["requests"] < 0).any():
    raise ValueError("Negative request counts")

df = df.sort_values(["date", "department"]).reset_index(drop=True)
departments = sorted(df["department"].unique().tolist())

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
    return features

train = df[df["date"] < "2025-10-01"].copy()
val = df[(df["date"] >= "2025-10-01") & (df["date"] < "2026-04-01")].copy()
if train.empty or val.empty:
    raise ValueError("Train or validation period is empty")

x_train_df = make_features(train)
x_val_df = make_features(val)
feature_columns = x_train_df.columns.tolist()

y_mean = float(train["requests"].mean())
y_std = float(train["requests"].std(ddof=0))
if y_std < 1e-8:
    y_std = 1.0

x_train = torch.tensor(x_train_df.to_numpy(dtype="float32"))
y_train = torch.tensor((train["requests"].to_numpy(dtype="float32") - y_mean) / y_std)
x_val = torch.tensor(x_val_df.to_numpy(dtype="float32"))
y_val = val["requests"].to_numpy(dtype="float32")

class CalendarMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

model = CalendarMLP(len(feature_columns))
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
criterion = nn.MSELoss()

train_groups = train.assign(
    month=train["date"].dt.month,
    weekend=train["date"].dt.dayofweek >= 5,
)
val_groups = val.assign(
    month=val["date"].dt.month,
    weekend=val["date"].dt.dayofweek >= 5,
)
seasonal_means = train_groups.groupby(
    ["department", "month", "weekend"]
)["requests"].mean()
baseline_preds = np.array([
    seasonal_means.loc[(row.department, row.month, row.weekend)]
    for row in val_groups.itertuples()
], dtype="float32")
baseline_mae = float(np.mean(np.abs(baseline_preds - y_val)))

best_mae = float("inf")
best_epoch = 0
bad_epochs = 0

for epoch in range(1, 201):
    model.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val).numpy() * y_std + y_mean
    val_pred = np.maximum(val_pred, 0.0)
    val_mae = float(np.mean(np.abs(val_pred - y_val)))

    if val_mae < best_mae - 1e-5:
        best_mae = val_mae
        best_epoch = epoch
        bad_epochs = 0
        torch.save(model.state_dict(), OUT / "calendar_mlp.pt")
    else:
        bad_epochs += 1

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | validation MAE: {val_mae:.3f}")
    if bad_epochs >= 25:
        print("Early stopping")
        break

config = {
    "departments": departments,
    "feature_columns": feature_columns,
    "target_mean": y_mean,
    "target_std": y_std,
    "architecture": [len(feature_columns), 64, 32, 1],
    "best_epoch": best_epoch,
    "validation_mae": best_mae,
    "seasonal_baseline_validation_mae": baseline_mae,
    "train_end": str(train["date"].max().date()),
    "val_end": str(val["date"].max().date()),
    "training_note": "Calendar-only model: no request history, no future target leakage; assumes repeating seasonality",
}
with open(OUT / "calendar_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("\n=== VALIDATION RESULTS ===")
print("Training examples:", len(train))
print("Validation examples:", len(val))
print("Features:", len(feature_columns))
print("Best epoch:", best_epoch)
print(f"Calendar MLP MAE: {best_mae:.3f}")
print(f"Seasonal baseline MAE: {baseline_mae:.3f}")
print("Model saved:", OUT / "calendar_mlp.pt")
print("Config saved:", OUT / "calendar_config.json")

if "Приемная комиссия" in departments:
    days = pd.date_range("2027-07-01", "2027-07-31", freq="D")
    example = pd.DataFrame({
        "date": days,
        "department": ["Приемная комиссия"] * len(days),
    })
    example_x = torch.tensor(make_features(example)[feature_columns].to_numpy(dtype="float32"))
    best_model = CalendarMLP(len(feature_columns))
    best_model.load_state_dict(torch.load(OUT / "calendar_mlp.pt", map_location="cpu", weights_only=True))
    best_model.eval()
    with torch.inference_mode():
        example_pred = best_model(example_x).numpy() * y_std + y_mean
    example_pred = np.maximum(example_pred, 0.0)
    print(f"\nJuly 2027 admission office (sanity check): {example_pred.mean():.2f} requests/day")
    print("Note: long-term calendar scenario, not verified future accuracy")
