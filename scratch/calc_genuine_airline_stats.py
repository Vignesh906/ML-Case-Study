import os
import pandas as pd
import numpy as np

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
flights_path = os.path.join(BASE_DIR, "archive", "flights.csv")

print("Calculating genuine airline delay statistics from archive/flights.csv...")
chunks = []
for chunk in pd.read_csv(flights_path, usecols=['MONTH', 'AIRLINE', 'ORIGIN_AIRPORT', 'DEPARTURE_DELAY'], chunksize=500000):
    # Only training months 1..9 to avoid test leakage
    chunk_tr = chunk[chunk['MONTH'] <= 9].dropna(subset=['DEPARTURE_DELAY'])
    chunks.append(chunk_tr)

df_all = pd.concat(chunks)
print(f"Total training split flights (Months 1-9): {len(df_all):,}")

# 1. Raw departure delay mean and median by airline
airline_stats = df_all.groupby('AIRLINE')['DEPARTURE_DELAY'].agg(
    count='count',
    raw_mean='mean',
    raw_median='median',
    pct_negative=lambda x: (x < 0).mean() * 100,
    pct_zero=lambda x: (x == 0).mean() * 100,
    clipped_mean=lambda x: x.clip(lower=0.0).mean(),
    clipped_median=lambda x: x.clip(lower=0.0).median()
).reset_index()

print("\n=== Genuine Airline Delay Statistics (Months 1-9) ===")
print(airline_stats.to_string(index=False))

# 2. Overall statistics
print(f"\nOverall Raw Mean: {df_all['DEPARTURE_DELAY'].mean():.2f}")
print(f"Overall Raw Median: {df_all['DEPARTURE_DELAY'].median():.2f}")
print(f"Overall Clipped Mean: {df_all['DEPARTURE_DELAY'].clip(lower=0.0).mean():.2f}")
print(f"Overall Clipped Median: {df_all['DEPARTURE_DELAY'].clip(lower=0.0).median():.2f}")

# 3. Check top 15 origin airports
airport_stats = df_all.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].agg(
    count='count',
    clipped_mean=lambda x: x.clip(lower=0.0).mean(),
    clipped_median=lambda x: x.clip(lower=0.0).median()
).sort_values(by='count', ascending=False).head(15).reset_index()

print("\n=== Top 15 Origin Airports Genuine Delay (Months 1-9) ===")
print(airport_stats.to_string(index=False))
