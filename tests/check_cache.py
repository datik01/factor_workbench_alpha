import pandas as pd
df = pd.read_parquet(".cache/universe_3176_2021_2026_20260417.parquet")
print("COLUMNS:")
print(df.columns.tolist())
print("\nHEAD(3):")
print(df.head(3).to_string())
