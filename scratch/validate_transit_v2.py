import os
import pandas as pd
import numpy as np

DATA_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study\transit_v2"
train_path = os.path.join(DATA_DIR, "train.csv")
val_path = os.path.join(DATA_DIR, "val.csv")
test_path = os.path.join(DATA_DIR, "test (1).csv")

print("--- Validating transit_v2 Datasets ---")
train = pd.read_csv(train_path, low_memory=False)
val = pd.read_csv(val_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)

print(f"Train rows: {len(train):,}, cols: {len(train.columns)}")
print(f"Val rows: {len(val):,}, cols: {len(val.columns)}")
print(f"Test rows: {len(test):,}, cols: {len(test.columns)}")

for split_name, df in [('Train', train), ('Val', val), ('Test', test)]:
    print(f"\n=== Split: {split_name} ===")
    for mode in ['Bus', 'Rail', 'Metro']:
        sub = df[df['route_type'].str.lower() == mode.lower()]
        delays = sub['delay_minutes']
        on_time = (delays <= 5).mean()
        moderate = ((delays > 5) & (delays <= 20)).mean()
        severe = (delays > 20).mean()
        p_zero = (delays == 0).mean()
        p_gt0 = (delays > 0).mean()
        p_gt5 = (delays > 5).mean()
        mean_d = delays.mean()
        median_d = delays.median()
        p90_d = np.percentile(delays, 90)
        
        print(f"Mode: {mode:5s} (N={len(sub):,}): "
              f"Mean={mean_d:.2f}m, Median={median_d:.1f}m, P90={p90_d:.1f}m | "
              f"On-Time={on_time*100:.1f}%, Moderate={moderate*100:.1f}%, Severe={severe*100:.1f}% | "
              f"P(d=0)={p_zero*100:.1f}%, P(d>0)={p_gt0*100:.1f}%, P(d>5)={p_gt5*100:.1f}%")

print("\n=== Generator Validation Tables (Train Split) ===")
# By Weather
for mode in ['Bus', 'Rail', 'Metro']:
    sub = train[train['route_type'].str.lower() == mode.lower()]
    print(f"\n--- {mode} by Weather ---")
    agg = sub.groupby('weather_condition')['delay_minutes'].agg(
        Count='count',
        Mean='mean',
        P_gt0=lambda x: (x > 0).mean(),
        P_gt5=lambda x: (x > 5).mean(),
        P_gt20=lambda x: (x > 20).mean()
    )
    print(agg.round(3))

# By Hour Band
train['hour_band'] = pd.cut(train['hour'], bins=[-1, 6, 9, 15, 19, 24], labels=['LateNight/Early', 'MorningPeak', 'Midday', 'EveningPeak', 'Night'])
for mode in ['Bus', 'Rail', 'Metro']:
    sub = train[train['route_type'].str.lower() == mode.lower()]
    print(f"\n--- {mode} by Hour Band ---")
    agg = sub.groupby('hour_band', observed=False)['delay_minutes'].agg(
        Count='count',
        Mean='mean',
        P_gt0=lambda x: (x > 0).mean(),
        P_gt5=lambda x: (x > 5).mean(),
        P_gt20=lambda x: (x > 20).mean()
    )
    print(agg.round(3))

# By Occupancy Band
train['occ_band'] = pd.cut(train['occupancy_rate'], bins=[-0.1, 0.4, 0.7, 0.85, 1.1], labels=['Low (<0.4)', 'Moderate (0.4-0.7)', 'High (0.7-0.85)', 'Crush (>0.85)'])
for mode in ['Bus', 'Rail', 'Metro']:
    sub = train[train['route_type'].str.lower() == mode.lower()]
    print(f"\n--- {mode} by Occupancy Band ---")
    agg = sub.groupby('occ_band', observed=False)['delay_minutes'].agg(
        Count='count',
        Mean='mean',
        P_gt0=lambda x: (x > 0).mean(),
        P_gt5=lambda x: (x > 5).mean(),
        P_gt20=lambda x: (x > 20).mean()
    )
    print(agg.round(3))
