import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prediction import DelayPredictor
import joblib

def audit_metro():
    print("=" * 80)
    print("AUDITING METRO PIPELINE & DATA QUALITY")
    print("=" * 80)

    # 1. Inspect original transit/train.csv for Metro
    orig_train = os.path.join(BASE_DIR, "transit", "train.csv")
    if os.path.exists(orig_train):
        df_orig = pd.read_csv(orig_train, low_memory=False)
        metro_orig = df_orig[df_orig['route_type'].astype(str).str.lower() == 'metro']
        print(f"\n1. ORIGINAL TRANSIT (transit/train.csv) METRO STATS:")
        print(f"   Rows: {len(metro_orig):,}, Columns: {len(metro_orig.columns)}")
        delays_orig = pd.to_numeric(metro_orig['delay_minutes'], errors='coerce').dropna().values
        print(f"   Delay % Zero: {np.mean(delays_orig == 0)*100:.2f}%")
        print(f"   Delay Mean: {np.mean(delays_orig):.4f} min, Std: {np.std(delays_orig):.4f}")
        print(f"   Median: {np.median(delays_orig):.4f}, P50: {np.percentile(delays_orig, 50):.4f}, P90: {np.percentile(delays_orig, 90):.4f}, P95: {np.percentile(delays_orig, 95):.4f}")
        cats_orig = pd.cut(delays_orig, bins=[-1, 5, 20, 1000], labels=['On Time (<=5m)', 'Moderate (6-20m)', 'Severe (>20m)'])
        print("   Class Proportions (Original):")
        print(pd.Series(cats_orig).value_counts(normalize=True).round(4) * 100)

    # 2. Inspect transit_v2/train.csv for Metro
    v2_train = os.path.join(BASE_DIR, "transit_v2", "train.csv")
    if os.path.exists(v2_train):
        df_v2 = pd.read_csv(v2_train, low_memory=False)
        metro_v2 = df_v2[df_v2['route_type'].astype(str).str.lower() == 'metro']
        print(f"\n2. TRANSIT V2 BENCHMARK (transit_v2/train.csv) METRO STATS:")
        print(f"   Rows: {len(metro_v2):,}, Columns: {len(metro_v2.columns)}")
        delays_v2 = pd.to_numeric(metro_v2['delay_minutes'], errors='coerce').dropna().values
        print(f"   Delay % Zero: {np.mean(delays_v2 == 0)*100:.2f}%")
        print(f"   Delay Mean: {np.mean(delays_v2):.4f} min, Std: {np.std(delays_v2):.4f}")
        print(f"   Median: {np.median(delays_v2):.4f}, P50: {np.percentile(delays_v2, 50):.4f}, P90: {np.percentile(delays_v2, 90):.4f}, P95: {np.percentile(delays_v2, 95):.4f}")
        cats_v2 = pd.cut(delays_v2, bins=[-1, 5, 20, 1000], labels=['On Time (<=5m)', 'Moderate (6-20m)', 'Severe (>20m)'])
        print("   Class Proportions (v2):")
        print(pd.Series(cats_v2).value_counts(normalize=True).round(4) * 100)

        # Check operational distributions
        print("\n   Delay by Headway in v2:")
        print(metro_v2.groupby(pd.cut(metro_v2['headway_min'], bins=[-1, 4, 8, 15, 30]), observed=False)['delay_minutes'].agg(['count', 'mean', lambda x: (x > 5).mean()*100]).round(2))
        print("\n   Delay by Peak Hour in v2:")
        print(metro_v2.groupby('is_peak_hour')['delay_minutes'].agg(['count', 'mean', lambda x: (x > 5).mean()*100]).round(2))
        print("\n   Delay by Occupancy in v2:")
        print(metro_v2.groupby(pd.cut(metro_v2['occupancy_rate'], bins=[-0.1, 0.4, 0.7, 0.88, 1.01]), observed=False)['delay_minutes'].agg(['count', 'mean', lambda x: (x > 5).mean()*100]).round(2))

    # 3. Inspect processed multimodal training and test data for METRO
    proc_train = os.path.join(BASE_DIR, "data", "processed", "multimodal_train.csv")
    proc_test = os.path.join(BASE_DIR, "data", "processed", "multimodal_test.csv")
    df_ptr = pd.read_csv(proc_train)
    df_pts = pd.read_csv(proc_test)

    metro_ptr = df_ptr[df_ptr['transport_mode'] == 'METRO']
    metro_pts = df_pts[df_pts['transport_mode'] == 'METRO']
    print(f"\n3. PROCESSED MULTIMODAL METRO STATS:")
    print(f"   Train samples: {len(metro_ptr):,}, Test samples: {len(metro_pts):,}")
    print(f"   Train % Zero: {np.mean(metro_ptr['delay_minutes'] == 0)*100:.2f}%, Test % Zero: {np.mean(metro_pts['delay_minutes'] == 0)*100:.2f}%")
    print(f"   Train Mean: {metro_ptr['delay_minutes'].mean():.4f} min, Test Mean: {metro_pts['delay_minutes'].mean():.4f} min")
    print(f"   Train Classes: {metro_ptr['delay_category'].value_counts(normalize=True).to_dict()}")
    print(f"   Test Classes: {metro_pts['delay_category'].value_counts(normalize=True).to_dict()}")

    # 4. Check Metro Regressor Artifact
    reg_dict_path = os.path.join(BASE_DIR, "models", "delay_regressors.joblib")
    reg_dict = joblib.load(reg_dict_path)
    metro_reg = reg_dict.get('METRO')
    print(f"\n4. METRO REGRESSOR ARTIFACT:")
    print(f"   Type: {type(metro_reg)}")
    print(f"   Params: max_iter={metro_reg.max_iter}, min_samples_leaf={metro_reg.min_samples_leaf}, lr={metro_reg.learning_rate}")

    # 5. Check DelayPredictor inference behavior for METRO
    predictor = DelayPredictor()
    print("\n5. METRO INFERENCE SENSITIVITY CHECK:")
    base_metro = {
        'transport_mode': 'METRO', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_001',
        'origin': 'STN_BER_0001', 'destination': 'STN_BER_0015',
        'scheduled_duration_min': 22.0, 'distance_km': 14.0,
        'historical_delay_mean': 0.4, 'hour': 12, 'day_of_week': 1, 'month': 10,
        'vehicle_capacity': 600.0, 'occupancy_rate': 0.50, 'headway_min': 4.0,
        'temperature_c': 15.9, 'weather_condition': 'clear'
    }

    scenarios = [
        ("Base Nominal (Midday hr 12, occ 0.50, hw 4m, clear)", base_metro),
        ("Headway Spacing Variation (hw 12m)", {**base_metro, 'headway_min': 12.0}),
        ("Headway Spacing Severe (hw 25m)", {**base_metro, 'headway_min': 25.0}),
        ("Occupancy Crowding (occ 0.75)", {**base_metro, 'occupancy_rate': 0.75}),
        ("Occupancy Overcrowded (occ 0.95)", {**base_metro, 'occupancy_rate': 0.95}),
        ("Peak Morning Rush (hr 8, occ 0.85, hw 6m)", {**base_metro, 'hour': 8, 'occupancy_rate': 0.85, 'headway_min': 6.0}),
        ("Peak Evening Rush (hr 18, occ 0.92, hw 8m)", {**base_metro, 'hour': 18, 'occupancy_rate': 0.92, 'headway_min': 8.0}),
        ("Historical Corridor Prior (hist 2.5m)", {**base_metro, 'historical_delay_mean': 2.5}),
        ("Adverse Weather Test (storm, 15.9C)", {**base_metro, 'weather_condition': 'storm'}),
        ("Subzero Cold Test (-5C, clear)", {**base_metro, 'temperature_c': -5.0}),
        ("High Heat Test (38C, clear)", {**base_metro, 'temperature_c': 38.0}),
        ("Route Duration Variation (sched 45m)", {**base_metro, 'scheduled_duration_min': 45.0}),
        ("Full Service Disruption (hr 18, occ 0.98, hw 25m, hist 5.0m)", {**base_metro, 'hour': 18, 'occupancy_rate': 0.98, 'headway_min': 25.0, 'historical_delay_mean': 5.0})
    ]

    for label, payload in scenarios:
        res = predictor.predict(payload)
        d = res['estimated_delay_min']
        cat = res['predicted_category']
        probs = res['probabilities']
        p50 = res.get('typical_delay_p50_min')
        p90 = res.get('high_end_delay_p90_min')
        print(f"\n   * {label}:")
        print(f"     Est Delay: {d} min | P50: {p50} min | P90: {p90} min | Category: {cat}")
        print(f"     Probabilities: On Time={probs['ON TIME']}%, Moderate={probs['MODERATELY DELAYED']}%, Severe={probs['SEVERELY DELAYED']}%")
        print(f"     Disagreement: {res.get('has_disagreement')}")

if __name__ == "__main__":
    audit_metro()
