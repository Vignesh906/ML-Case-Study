"""
Validation Script for Synthetic Urban Transit Simulation Benchmark v2
Computes empirical mean delay and P(delay > 5) across:
- Weather conditions
- Hour bands (Early Morning, Morning Rush, Midday, Evening Rush, Night)
- Occupancy bands (<0.4, 0.4-0.7, 0.7-0.9, >0.9)
- Headway bands (Low <=8m, Medium 9-15m, High >15m)
- Temperature bands (Freezing <0C, Mild 0-25C, Hot >25C)
Per mode (Bus, Rail, Metro), and verifies target class proportions.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
DATA_PATH = os.path.join(BASE_DIR, "transit_v2", "train.csv")

def validate_transit_v2():
    print(f"Loading {DATA_PATH} for validation...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df['route_type'] = df['route_type'].astype(str).str.upper()
    modes = ['BUS', 'RAIL', 'METRO']

    print("=" * 80)
    print("TRANSIT V2 BENCHMARK VALIDATION REPORT")
    print("=" * 80)

    # 1. Class Distribution per mode
    print("\n--- 1. DELAY SEVERITY CLASS PROPORTIONS ---")
    for m in modes:
        sub = df[df['route_type'] == m]
        cats = pd.cut(sub['delay_minutes'], bins=[-1, 5, 20, 1000], labels=['ON TIME (<=5m)', 'MODERATE (6-20m)', 'SEVERE (>20m)'])
        dist = cats.value_counts(normalize=True) * 100
        print(f"\nMode: {m} (n = {len(sub):,} rows)")
        for cat_name, pct in dist.items():
            cnt = (cats == cat_name).sum()
            print(f"  {cat_name:<20}: {pct:5.2f}% ({cnt:,} trips)")

    # 2. Weather Sensitivity per mode
    print("\n" + "=" * 80)
    print("--- 2. WEATHER SENSITIVITY ---")
    for m in modes:
        sub = df[df['route_type'] == m]
        print(f"\nMode: {m}")
        wx_agg = sub.groupby('weather_condition')['delay_minutes'].agg(
            count='count',
            mean_delay='mean',
            median_delay='median',
            pct_over_5m=lambda x: (x > 5).mean() * 100,
            pct_over_20m=lambda x: (x > 20).mean() * 100
        ).loc[['clear', 'cloudy', 'fog', 'light_rain', 'heavy_rain', 'snow', 'storm']]
        print(wx_agg.round(2))

    # 3. Peak Hour vs Off-Peak
    print("\n" + "=" * 80)
    print("--- 3. PEAK HOUR SENSITIVITY ---")
    for m in modes:
        sub = df[df['route_type'] == m]
        print(f"\nMode: {m}")
        pk_agg = sub.groupby('is_peak_hour')['delay_minutes'].agg(
            count='count',
            mean_delay='mean',
            pct_over_5m=lambda x: (x > 5).mean() * 100
        )
        pk_agg.index = ['Off-Peak (Hour)', 'Peak Rush (Hour)']
        print(pk_agg.round(2))

    # 4. Occupancy Bands
    print("\n" + "=" * 80)
    print("--- 4. OCCUPANCY RATE SENSITIVITY ---")
    occ_bins = [0.0, 0.40, 0.70, 0.88, 1.01]
    occ_labels = ['Low (<40%)', 'Nominal (40-70%)', 'Crowded (70-88%)', 'Overcrowded (>88%)']
    df['occ_band'] = pd.cut(df['occupancy_rate'], bins=occ_bins, labels=occ_labels)

    for m in modes:
        sub = df[df['route_type'] == m]
        print(f"\nMode: {m}")
        occ_agg = sub.groupby('occ_band', observed=False)['delay_minutes'].agg(
            count='count',
            mean_delay='mean',
            pct_over_5m=lambda x: (x > 5).mean() * 100
        )
        print(occ_agg.round(2))

    # 5. Headway Bands
    print("\n" + "=" * 80)
    print("--- 5. HEADWAY BAND SENSITIVITY ---")
    hw_bins = [-1.0, 5.0, 12.0, 25.0, 120.0]
    hw_labels = ['Frequent (<=5m)', 'Standard (6-12m)', 'Infrequent (13-25m)', 'Long Headway (>25m)']
    df['hw_band'] = pd.cut(df['headway_min'], bins=hw_bins, labels=hw_labels)

    for m in modes:
        sub = df[df['route_type'] == m]
        print(f"\nMode: {m}")
        hw_agg = sub.groupby('hw_band', observed=False)['delay_minutes'].agg(
            count='count',
            mean_delay='mean',
            pct_over_5m=lambda x: (x > 5).mean() * 100
        )
        print(hw_agg.round(2))

    # 6. Temperature Bands
    print("\n" + "=" * 80)
    print("--- 6. TEMPERATURE BAND SENSITIVITY ---")
    temp_bins = [-50.0, 0.0, 30.0, 60.0]
    temp_labels = ['Freezing (<=0C)', 'Mild (0-30C)', 'Extreme Heat (>30C)']
    df['temp_band'] = pd.cut(df['temperature_c'], bins=temp_bins, labels=temp_labels)

    for m in modes:
        sub = df[df['route_type'] == m]
        print(f"\nMode: {m}")
        temp_agg = sub.groupby('temp_band', observed=False)['delay_minutes'].agg(
            count='count',
            mean_delay='mean',
            pct_over_5m=lambda x: (x > 5).mean() * 100
        )
        print(temp_agg.round(2))

if __name__ == "__main__":
    validate_transit_v2()
