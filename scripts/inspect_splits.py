import pandas as pd
from pathlib import Path

csv_path = Path("ml/results/phase_8_5/phase_8_5_test_predictions.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print("Total rows in phase_8_5_test_predictions.csv:", len(df))
    print("Strategy counts:\n", df['strategy_id'].value_counts())

for p in Path("ml/results").glob("**/*test*.csv"):
    df_temp = pd.read_csv(p)
    print(f"{p}: rows={len(df_temp)}, columns={list(df_temp.columns[:5])}")

for p in Path("ml/results").glob("**/*test*.json"):
    print(f"JSON: {p}")
