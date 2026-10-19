
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

departments = {
    "Учебный отдел": {
        "base": 18,
        "categories": ["Справки", "Расписание", "Академические вопросы"],
        "peaks": {2: 1.5, 9: 1.7, 7: 0.5},
    },
    "Бухгалтерия": {
        "base": 10,
        "categories": ["Оплата", "Задолженность", "Возврат"],
        "peaks": {2: 1.8, 9: 1.9},
    },
    "Приемная комиссия": {
        "base": 5,
        "categories": ["Поступление", "Документы", "Зачисление"],
        "peaks": {6: 3.0, 7: 4.0, 8: 3.0},
    },
    "Общежития": {
        "base": 8,
        "categories": ["Заселение", "Ремонт", "Проживание"],
        "peaks": {8: 2.5, 9: 3.0},
    },
    "IT-поддержка": {
        "base": 15,
        "categories": ["Учетная запись", "Wi-Fi", "Учебная система"],
        "peaks": {9: 1.8, 2: 1.4, 7: 0.6},
    },
}

dates = pd.date_range("2023-09-01", "2026-09-30")
records = []
request_id = 1

for date in dates:
    for department, config in departments.items():

        month_factor = config["peaks"].get(date.month, 1.0)
        weekend_factor = (
            0.45 if department == "Общежития" else 0.25
        ) if date.weekday() >= 5 else 1.0

        trend = 1 + 0.03 * (date.year - 2023)
        random_factor = rng.gamma(shape=12, scale=1 / 12)

        expected = (
            config["base"]
            * month_factor
            * weekend_factor
            * trend
            * random_factor
        )

        count = rng.poisson(expected)

        for _ in range(count):
            minute = int(rng.integers(8 * 60, 20 * 60))
            created_at = date + pd.Timedelta(minutes=minute)
            category = rng.choice(config["categories"])

            records.append({
                "request_id": f"REQ-{request_id:07d}",
                "created_at": created_at,
                "department": department,
                "category": category,
                "data_source": "synthetic",
            })

            request_id += 1

df = pd.DataFrame(records)
df = df.sort_values("created_at").reset_index(drop=True)

output = Path("data/university_requests_synthetic.csv")
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)

print("Файл создан:", output)
print("Всего обращений:", len(df))
print("Начало:", df["created_at"].min())
print("Конец:", df["created_at"].max())

print("\nРаспределение по подразделениям:")
print(df["department"].value_counts())

print("\nПервые пять записей:")
print(df.head().to_string(index=False))
