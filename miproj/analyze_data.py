
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Загружаем созданный датасет
path = Path("data/university_requests_synthetic.csv")
df = pd.read_csv(path, parse_dates=["created_at"])

print("\n=== ОБЩАЯ ИНФОРМАЦИЯ ===")
print("Количество обращений:", len(df))
print("Количество столбцов:", len(df.columns))
print("Столбцы:", df.columns.tolist())

print("\nПериод данных:")
print(df["created_at"].min())
print(df["created_at"].max())

# 2. Проверяем качество данных
print("\n=== ПРОПУСКИ ===")
print(df.isna().sum())

print("\n=== ДУБЛИКАТЫ ===")
print("Повторяющиеся ID:", df["request_id"].duplicated().sum())
print("Полные дубликаты:", df.duplicated().sum())

# 3. Распределение обращений
print("\n=== ПОДРАЗДЕЛЕНИЯ ===")
print(df["department"].value_counts())

print("\n=== КАТЕГОРИИ ===")
print(df["category"].value_counts())

# 4. Подсчитываем обращения за каждый день
df["date"] = df["created_at"].dt.normalize()

daily = (
    df.groupby(["date", "department"])
    .size()
    .unstack(fill_value=0)
)

# Добавляем все календарные дни, включая дни без обращений
all_dates = pd.date_range(
    df["date"].min(),
    df["date"].max(),
    freq="D"
)
daily = daily.reindex(all_dates, fill_value=0)

print("\n=== СРЕДНЯЯ НАГРУЗКА В ДЕНЬ ===")
print(daily.mean().round(2))

print("\nКоличество календарных дней:", len(daily))

# 5. Создаём папку для графиков
charts = Path("charts")
charts.mkdir(exist_ok=True)

# График №1: обращения по месяцам
monthly = daily.resample("MS").sum()

ax = monthly.plot(figsize=(12, 5))
ax.set_title("Monthly requests by department")
ax.set_xlabel("Month")
ax.set_ylabel("Number of requests")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(charts / "monthly_requests.png", dpi=150)
plt.close()

# График №2: средняя нагрузка по дням недели
weekday = daily.groupby(daily.index.dayofweek).mean()

ax = weekday.plot(figsize=(10, 5), marker="o")
ax.set_title("Average daily requests by weekday")
ax.set_xlabel("Day of week")
ax.set_ylabel("Average requests")
ax.set_xticks(range(7))
ax.set_xticklabels([
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
])
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(charts / "weekday_requests.png", dpi=150)
plt.close()

print("\nГотово! Графики сохранены в папке charts.")
