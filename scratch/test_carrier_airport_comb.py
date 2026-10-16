import os
import pandas as pd
import numpy as np

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
flights_path = os.path.join(BASE_DIR, "archive", "flights.csv")

# Sample training flights (months 1-9)
chunks = []
for chunk in pd.read_csv(flights_path, usecols=['MONTH', 'AIRLINE', 'ORIGIN_AIRPORT', 'DEPARTURE_DELAY'], chunksize=500000):
    chunk_tr = chunk[chunk['MONTH'] <= 9].dropna(subset=['DEPARTURE_DELAY'])
    chunks.append(chunk_tr)
df_tr = pd.concat(chunks)

carrier_map = df_tr.groupby('AIRLINE')['DEPARTURE_DELAY'].apply(lambda s: s.clip(lower=0).mean()).to_dict()
airport_map = df_tr.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].apply(lambda s: s.clip(lower=0).mean()).to_dict()
overall_mean = float(df_tr['DEPARTURE_DELAY'].clip(lower=0).mean())

print("Carrier map (sample):", {k: round(v, 2) for k, v in list(carrier_map.items())[:5]})
print("Airport map (sample):", {k: round(v, 2) for k, v in list(airport_map.items())[:5]})

# Check combined baseline for different pairs
pairs = [
    ('DL', 'ATL'),
    ('DL', 'ORD'),
    ('DL', 'JFK'),
    ('DL', 'SFO'),
    ('DL', 'SEA'),
    ('UA', 'ORD'),
    ('UA', 'DEN'),
    ('HA', 'HNL'),
    ('NK', 'ORD'),
    ('NK', 'FLL'),
    ('AS', 'SEA')
]

print("\n=== Combined Baseline (0.5 * Carrier + 0.5 * Origin Airport) ===")
for al, ap in pairs:
    c_val = carrier_map.get(al, overall_mean)
    a_val = airport_map.get(ap, overall_mean)
    comb = 0.5 * c_val + 0.5 * a_val
    print(f"Airline: {al:<3} ({c_val:.2f}m) + Origin: {ap:<4} ({a_val:.2f}m) -> Combined: {comb:.2f}m")
