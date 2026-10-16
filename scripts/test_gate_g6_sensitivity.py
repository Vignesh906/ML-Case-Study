"""
Gate G6: One-At-A-Time Sensitivity Suite for All 4 Modes
--------------------------------------------------------
Evaluates effect of each feature on Expected Delay and P(Severe)
with raw unrounded values and verifies domain sensitivity consistency.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import MultimodalPreprocessor

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODES = ["BUS", "TRAIN", "METRO", "AIRPLANE"]
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))
mode_classifiers = joblib.load(os.path.join(MODELS_DIR, "mode_classifiers.joblib"))
mode_regressors = joblib.load(os.path.join(MODELS_DIR, "mode_regressors.joblib"))

def evaluate_point(row_dict):
    df = pd.DataFrame([row_dict])
    X = preprocessor.transform(df)
    mode = row_dict['transport_mode']
    
    clf = mode_classifiers[mode]
    reg = mode_regressors[mode]
    
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        p_sev = float(probs[2])
    else:
        p_sev = 0.0
        
    if hasattr(reg, "predict"):
        exp_d = float(reg.predict(X, p_sev=np.array([p_sev]))[0]) if 'p_sev' in reg.predict.__code__.co_varnames else float(reg.predict(X)[0])
    else:
        exp_d = 0.0
    return exp_d, p_sev

def run_mode_sensitivity(mode):
    print(f"\n==================== SENSITIVITY: {mode} ====================")
    
    # Baseline input
    if mode == 'BUS':
        base = {
            'transport_mode': 'BUS', 'hour': 12, 'day_of_week': 2, 'month': 6, 'day': 15,
            'is_weekend': 0, 'is_peak_hour': 0, 'distance_km': 25.0, 'scheduled_duration_min': 50.0,
            'carrier_or_route': 'RT_LON_001', 'origin': 'STN_001', 'destination': 'STN_002', 'city': 'London',
            'historical_delay_mean': 3.5, 'vehicle_capacity': 80.0, 'occupancy_rate': 0.50,
            'passenger_count': 40.0, 'weather_condition': 'clear', 'temperature_c': 18.0,
            'headway_min': 10.0, 'delay_minutes': 0.0
        }
    elif mode == 'TRAIN':
        base = {
            'transport_mode': 'TRAIN', 'hour': 12, 'day_of_week': 2, 'month': 6, 'day': 15,
            'is_weekend': 0, 'is_peak_hour': 0, 'distance_km': 75.0, 'scheduled_duration_min': 65.0,
            'carrier_or_route': 'RT_LON_RAIL_01', 'origin': 'STN_001', 'destination': 'STN_002', 'city': 'London',
            'historical_delay_mean': 4.0, 'vehicle_capacity': 600.0, 'occupancy_rate': 0.50,
            'passenger_count': 300.0, 'weather_condition': 'clear', 'temperature_c': 18.0,
            'headway_min': 20.0, 'delay_minutes': 0.0
        }
    elif mode == 'METRO':
        base = {
            'transport_mode': 'METRO', 'hour': 12, 'day_of_week': 2, 'month': 6, 'day': 15,
            'is_weekend': 0, 'is_peak_hour': 0, 'distance_km': 18.0, 'scheduled_duration_min': 30.0,
            'carrier_or_route': 'RT_LON_METRO_01', 'origin': 'STN_001', 'destination': 'STN_002', 'city': 'London',
            'historical_delay_mean': 1.5, 'vehicle_capacity': 800.0, 'occupancy_rate': 0.50,
            'passenger_count': 400.0, 'weather_condition': 'clear', 'temperature_c': 18.0,
            'headway_min': 5.0, 'delay_minutes': 0.0
        }
    else:  # AIRPLANE
        base = {
            'transport_mode': 'AIRPLANE', 'hour': 12, 'day_of_week': 2, 'month': 6, 'day': 15,
            'is_weekend': 0, 'is_peak_hour': 0, 'distance_km': 950.0, 'scheduled_duration_min': 130.0,
            'carrier_or_route': 'DL', 'origin': 'ATL', 'destination': 'LGA', 'city': 'ATL',
            'historical_delay_mean': 11.5, 'vehicle_capacity': 180.0, 'occupancy_rate': 0.82,
            'passenger_count': 148.0, 'weather_condition': 'clear', 'temperature_c': 18.0,
            'headway_min': 0.0, 'delay_minutes': 0.0
        }
        
    base_d, base_p = evaluate_point(base)
    print(f"Base Condition: Expected Delay = {base_d:.4f} min, P(Severe) = {base_p*100:.2f}%")
    
    results = []
    
    # Test suite variations
    tests = []
    # Weather
    for w in ['cloudy', 'fog', 'light_rain', 'heavy_rain', 'snow', 'storm']:
        tests.append(('Weather', w, {'weather_condition': w}))
    # Temp
    tests.append(('Temperature', '-5 C', {'temperature_c': -5.0}))
    tests.append(('Temperature', '38 C', {'temperature_c': 38.0}))
    # Hour
    tests.append(('Hour', '8 AM Peak', {'hour': 8, 'is_peak_hour': 1}))
    tests.append(('Hour', '17 PM Peak', {'hour': 17, 'is_peak_hour': 1}))
    tests.append(('Hour', '23 PM Night', {'hour': 23, 'is_peak_hour': 0}))
    # Occupancy
    tests.append(('Occupancy', 'Low (0.25)', {'occupancy_rate': 0.25}))
    tests.append(('Occupancy', 'High (0.85)', {'occupancy_rate': 0.85}))
    tests.append(('Occupancy', 'Crush (0.95)', {'occupancy_rate': 0.95}))
    # Headway
    tests.append(('Headway', 'Short (3 min)', {'headway_min': 3.0}))
    tests.append(('Headway', 'Long (25 min)', {'headway_min': 25.0}))
    # Capacity (Inert)
    tests.append(('Capacity', 'Double Capacity', {'vehicle_capacity': base['vehicle_capacity'] * 2.0}))
    # Historical Delay
    tests.append(('Hist Delay', 'High (15 min)', {'historical_delay_mean': 15.0}))
    # Distance / Duration
    tests.append(('Duration', 'Long Duration', {'scheduled_duration_min': base['scheduled_duration_min'] * 2.0}))
    
    if mode == 'AIRPLANE':
        tests.append(('Carrier', 'EV (High Delay Carrier)', {'carrier_or_route': 'EV'}))
        tests.append(('Carrier', 'HA (Low Delay Carrier)', {'carrier_or_route': 'HA'}))
        tests.append(('Origin', 'ORD (Chicago)', {'origin': 'ORD'}))
        tests.append(('Origin', 'PHX (Phoenix)', {'origin': 'PHX'}))
        
    for cat, label, mod in tests:
        # Check if weather option is unsupported for airplane
        if mode == 'AIRPLANE' and mod.get('weather_condition') in ['fog', 'storm']:
            continue
        p_dict = dict(base)
        p_dict.update(mod)
        d_val, p_val = evaluate_point(p_dict)
        delta_d = d_val - base_d
        delta_p = (p_val - base_p) * 100.0
        
        results.append({
            "mode": mode,
            "category": cat,
            "setting": label,
            "expected_delay": round(d_val, 4),
            "delta_delay": round(delta_d, 4),
            "p_severe_pct": round(p_val * 100.0, 2),
            "delta_p_severe_pts": round(delta_p, 2)
        })
        print(f"  {cat:12s} | {label:18s} -> Exp Delay: {d_val:7.4f}m (d: {delta_d:+7.4f}m) | P(Sev): {p_val*100:5.2f}% (d: {delta_p:+5.2f} pts)")
        
    return results

def main():
    all_res = []
    for mode in MODES:
        all_res.extend(run_mode_sensitivity(mode))
        
    out_path = os.path.join(BASE_DIR, "results", "metrics", "sensitivity_results.json")
    with open(out_path, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nAll sensitivity results saved to {out_path}")

if __name__ == "__main__":
    main()
