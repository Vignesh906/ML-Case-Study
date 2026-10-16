"""
Rerun Section 0 Test Cases a) to d) on the Updated Build
--------------------------------------------------------
Captures unrounded delay predictions, quantile estimates (P50 & P90),
probabilities, active models, and transformed features.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prediction import predictor_instance

def run_case(name: str, payload: dict) -> dict:
    # Run prediction
    res = predictor_instance.predict(payload)
    
    # Also extract unrounded raw regression value directly
    mode = payload.get('transport_mode', 'BUS')
    hour = payload.get('hour', 12)
    dow = payload.get('day_of_week', 1)
    month = payload.get('month', 10)
    day = payload.get('day', 15)
    
    row_dict = {
        'transport_mode': mode,
        'hour': hour,
        'day_of_week': dow,
        'month': month,
        'day': day,
        'is_weekend': 1 if dow in [5, 6] else 0,
        'is_peak_hour': 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0,
        'distance_km': float(payload.get('distance_km', 25.0)),
        'scheduled_duration_min': float(payload.get('scheduled_duration_min', 45.0)),
        'carrier_or_route': payload.get('carrier_or_route', 'RT_01'),
        'origin': payload.get('origin', 'ORIG'),
        'destination': payload.get('destination', 'DEST'),
        'city': payload.get('city', 'London'),
        'historical_delay_mean': float(payload.get('historical_delay_mean', 1.5)),
        'vehicle_capacity': float(payload.get('vehicle_capacity', 80.0)),
        'occupancy_rate': float(payload.get('occupancy_rate', 0.65)),
        'passenger_count': float(payload.get('passenger_count', 52.0)),
        'temperature_c': float(payload.get('temperature_c', 18.0)),
        'weather_condition': payload.get('weather_condition', 'clear'),
        'headway_min': float(payload.get('headway_min', 8.0 if mode != 'AIRPLANE' else 0.0))
    }
    df_single = pd.DataFrame([row_dict])
    X_vec = predictor_instance.preprocessor.transform(df_single)
    
    reg = predictor_instance.mode_regressors.get(mode)
    raw_unrounded_reg = float(reg.predict(X_vec)[0]) if reg is not None else None
    
    q50_reg = predictor_instance.mode_quantiles_p50.get(mode)
    raw_q50 = float(q50_reg.predict(X_vec)[0]) if q50_reg is not None else None
    
    q90_reg = predictor_instance.mode_quantiles_p90.get(mode)
    raw_q90 = float(q90_reg.predict(X_vec)[0]) if q90_reg is not None else None

    # Classifier raw decision
    clf = predictor_instance.best_model
    raw_proba = clf.predict_proba(X_vec)[0].tolist() if hasattr(clf, 'predict_proba') else None
    raw_clf_pred = int(clf.predict(X_vec)[0])

    return {
        "case_name": name,
        "payload": payload,
        "result_returned": res,
        "raw_unrounded_regression": raw_unrounded_reg,
        "raw_unrounded_p50": raw_q50,
        "raw_unrounded_p90": raw_q90,
        "raw_unrounded_probabilities": raw_proba,
        "raw_classifier_prediction": raw_clf_pred,
        "regressor_artifact": type(reg).__name__ if reg else "None",
        "classifier_artifact": type(clf).__name__ if clf else "None",
        "transformed_vector_dim": X_vec.shape[1],
        "nonzero_feature_count": int(np.count_nonzero(X_vec))
    }

def main():
    cases = {}

    # Case a: Bus London RT_LON_001
    bus_payload_base = {
        "transport_mode": "BUS",
        "city": "London",
        "carrier_or_route": "RT_LON_001",
        "origin": "STN_LON_0037",
        "destination": "STN_LON_0211",
        "hour": 8,
        "day_of_week": 1,
        "month": 10,
        "distance_km": 51.1,
        "scheduled_duration_min": 102.2,
        "historical_delay_mean": 1.2,
        "headway_min": 15,
        "vehicle_capacity": 80,
        "occupancy_rate": 0.65,
        "temperature_c": 16.8
    }
    
    bus_clear = dict(bus_payload_base, weather_condition="clear")
    bus_storm = dict(bus_payload_base, weather_condition="storm")
    cases["case_a_bus_clear"] = run_case("Bus London - Clear (16.8 C)", bus_clear)
    cases["case_a_bus_storm"] = run_case("Bus London - Severe Thunderstorm (16.8 C)", bus_storm)

    # Case b: Train Berlin RT_BER_002
    train_payload_base = {
        "transport_mode": "TRAIN",
        "city": "Berlin",
        "carrier_or_route": "RT_BER_002",
        "origin": "STN_BER_0001",
        "destination": "STN_BER_0005",
        "hour": 8,
        "day_of_week": 1,
        "month": 10,
        "distance_km": 40.0,
        "scheduled_duration_min": 50.0,
        "historical_delay_mean": 0.6,
        "headway_min": 20,
        "vehicle_capacity": 400,
        "occupancy_rate": 0.70
    }
    train_clear = dict(train_payload_base, weather_condition="clear", temperature_c=15.9)
    train_storm = dict(train_payload_base, weather_condition="storm", temperature_c=40.9)
    cases["case_b_train_clear"] = run_case("Train Berlin - Clear (15.9 C)", train_clear)
    cases["case_b_train_storm"] = run_case("Train Berlin - Severe Thunderstorm (40.9 C)", train_storm)

    # Case c: Metro Berlin RT_BER_001 (changing ONLY hour: 8 vs 12)
    metro_payload_base = {
        "transport_mode": "METRO",
        "city": "Berlin",
        "carrier_or_route": "RT_BER_001",
        "origin": "STN_BER_0001",
        "destination": "STN_BER_0005",
        "day_of_week": 1,
        "month": 10,
        "distance_km": 15.0,
        "scheduled_duration_min": 25.0,
        "historical_delay_mean": 0.4,
        "headway_min": 5,
        "vehicle_capacity": 800,
        "occupancy_rate": 0.75,
        "weather_condition": "clear",
        "temperature_c": 15.9
    }
    metro_hr8 = dict(metro_payload_base, hour=8)
    metro_hr12 = dict(metro_payload_base, hour=12)
    cases["case_c_metro_hour_8"] = run_case("Metro Berlin - Hour 8 (15.9 C)", metro_hr8)
    cases["case_c_metro_hour_12"] = run_case("Metro Berlin - Hour 12 (15.9 C)", metro_hr12)

    # Case d: Airplane DL ATL -> LAX, hour 12, Tuesday, October
    airplane_payload_base = {
        "transport_mode": "AIRPLANE",
        "city": "Atlanta",
        "carrier_or_route": "DL",
        "origin": "ATL",
        "destination": "LAX",
        "hour": 12,
        "day_of_week": 1,
        "month": 10,
        "distance_km": 3132.0,
        "scheduled_duration_min": 290.0,
        "historical_delay_mean": 10.9,
        "temperature_c": 21.7
    }
    airplane_clear = dict(airplane_payload_base, weather_condition="clear")
    airplane_storm = dict(airplane_payload_base, weather_condition="storm")
    cases["case_d_airplane_clear"] = run_case("Airplane DL ATL->LAX - Clear (21.7 C)", airplane_clear)
    cases["case_d_airplane_storm"] = run_case("Airplane DL ATL->LAX - Severe Thunderstorm (21.7 C)", airplane_storm)

    out_file = os.path.join(BASE_DIR, "scratch", "section_0_results_after.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print("=== Section 0 Test Cases After Fixes ===")
    for k, v in cases.items():
        res = v["result_returned"]
        reg_raw = v["raw_unrounded_regression"]
        q50_raw = v["raw_unrounded_p50"]
        q90_raw = v["raw_unrounded_p90"]
        probs = res["probabilities"]
        print(f"\n[{k}] {v['case_name']}:")
        print(f"  Estimated Delay (Expected): {res['estimated_delay_min']} min (raw unrounded: {reg_raw:.4f} min)")
        print(f"  Typical Median (P50): {res['typical_median_delay_min']} min (raw unrounded: {q50_raw:.4f} min)")
        print(f"  High-End Risk (P90): {res['high_end_p90_delay_min']} min (raw unrounded: {q90_raw:.4f} min)")
        print(f"  Predicted Category: {res['predicted_category']} (Risk: {res['risk_level']})")
        print(f"  Probabilities: On Time={probs['ON TIME']}%, Moderate={probs['MODERATELY DELAYED']}%, Severe={probs['SEVERELY DELAYED']}%")
        print(f"  Model Used: {res['classifier_name']} + {res['regressor_name']}")

if __name__ == "__main__":
    main()
