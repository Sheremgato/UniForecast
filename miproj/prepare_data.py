
from pathlib import Path
import pandas as pd

df = pd.read_csv(
    "data/university_requests_synthetic.csv",
    parse_dates=["created_at"]
)

required = ["request_id", "created_at", "department"]

assert not df[required].isna().any().any(), "Есть пропуски!"
assert df["request_id"].is_unique, "Есть повторяющиеся ID!"

total_original = len(df)

df["date"] = df["created_at"].dt.normalize()

counts = (
    df.groupby(["date", "department"])
    .size()
)

dates = pd.date_range(
    df["date"].min(),
    df["date"].max(),
    freq="D"
)

departments = sorted(df["department"].unique())

full_index = pd.MultiIndex.from_product(
    [dates, departments],
    names=["date", "department"]
)

daily = (
    counts.reindex(full_index, fill_value=0)
    .rename("requests")
    .reset_index()
)

total_daily = int(daily["requests"].sum())

assert total_original == total_daily, (
    "Количество обращений не совпадает!"
)

output = Path("data/daily_requests.csv")
daily.to_csv(output, index=False)

print("=== ДАННЫЕ ПОДГОТОВЛЕНЫ ===")
print("Подразделений:", len(departments))
print("Дней:", len(dates))
print("Строк:", len(daily))

print("Исходных обращений:", total_original)
print("После группировки:", total_daily)

print("\nПервые 10 строк:")
print(daily.head(10).to_string(index=False))

print("\nДней с нулевой нагрузкой по подразделениям:")
print(
    daily.groupby("department")["requests"]
    .apply(lambda x: (x == 0).sum())
)

print("\nФайл сохранён:", output)
