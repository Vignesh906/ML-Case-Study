"""
Extract Authentic 2015 US DOT Flights with Verified Open-Meteo Weather
----------------------------------------------------------------------
Chronological split:
- Train: Months 1-9 (240,000 flights, ~26,667 per month)
- Val: Month 10 (30,000 flights)
- Test: Months 11-12 (30,000 flights, 15,000 per month)
Total: 300,000 authentic flights.

Features strictly known prior to departure:
- AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, MONTH, DAY_OF_WEEK, DAY,
  SCHEDULED_DEPARTURE (hour), DISTANCE, SCHEDULED_TIME.
- Historical delay rates computed strictly from the TRAINING SPLIT ONLY.
- Weather joined from Open-Meteo archive for 25 verified airports;
  unmatched rows are DROPPED (never default to 'clear').
- No post-flight or leak columns.
"""

import os
import json
import time
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
FLIGHTS_CSV = os.path.join(BASE_DIR, "archive", "flights.csv")
WEATHER_JSON = os.path.join(BASE_DIR, "data", "historical_airport_weather_2015.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SEED = 42

DOT_TO_IATA = {
    '10397': 'ATL', '13930': 'ORD', '11298': 'DFW', '11292': 'DEN', '12892': 'LAX',
    '14771': 'SFO', '12266': 'IAH', '14107': 'PHX', '12889': 'LAS', '13487': 'MSP',
    '14747': 'SEA', '10721': 'BOS', '11433': 'DTW', '11618': 'EWR', '11057': 'CLT',
    '13204': 'MCO', '14869': 'SLC', '12953': 'LGA', '12478': 'JFK', '10821': 'BWI',
    '13232': 'MDW', '11278': 'DCA', '14679': 'SAN', '14100': 'PHL', '13303': 'MIA',
    '11697': 'FLL'
}

def load_verified_weather():
    with open(WEATHER_JSON, "r") as f:
        data = json.load(f)
    print(f"Loaded verified weather archive for {len(data)} airports.")
    return data

def extract_and_process_flights():
    start_t = time.time()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    weather_db = load_verified_weather()
    verified_airports = set(weather_db.keys())

    # Pre-departure columns only
    cols = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'DISTANCE',
        'CANCELLED', 'DIVERTED'
    ]

    target_per_month = {
        1: 26667, 2: 26667, 3: 26667, 4: 26667, 5: 26667,
        6: 26667, 7: 26667, 8: 26667, 9: 26667,
        10: 30000,
        11: 15000, 12: 15000
    }
    collected = {m: [] for m in range(1, 13)}

    print(f"Streaming {FLIGHTS_CSV}...")
    chunk_idx = 0
    for chunk in pd.read_csv(FLIGHTS_CSV, usecols=cols, chunksize=500000, low_memory=False):
        chunk_idx += 1
        # Normalize DOT numeric airport IDs to 3-letter IATA codes
        chunk['ORIGIN_AIRPORT'] = chunk['ORIGIN_AIRPORT'].astype(str).map(lambda x: DOT_TO_IATA.get(x, x))
        chunk['DESTINATION_AIRPORT'] = chunk['DESTINATION_AIRPORT'].astype(str).map(lambda x: DOT_TO_IATA.get(x, x))
        # 1. Drop cancelled, diverted, missing delay
        c1 = chunk[(chunk['CANCELLED'] == 0) & (chunk['DIVERTED'] == 0) & (chunk['DEPARTURE_DELAY'].notna())]
        # 2. Origin airport must be in verified list (strictly drop unmatched; no default to clear)
        c2 = c1[c1['ORIGIN_AIRPORT'].isin(verified_airports)]

        for m in range(1, 13):
            sub_m = c2[c2['MONTH'] == m]
            needed = target_per_month[m] - sum(len(x) for x in collected[m])
            if needed > 0 and len(sub_m) > 0:
                take = min(needed, len(sub_m))
                collected[m].append(sub_m.sample(n=take, random_state=SEED + m))

        done_cnt = sum(sum(len(x) for x in collected[m]) >= target_per_month[m] for m in range(1, 13))
        if done_cnt == 12:
            print(f"All monthly quotas filled by chunk {chunk_idx}!")
            break

    # Assemble raw splits
    train_chunks = [pd.concat(collected[m], ignore_index=True) for m in range(1, 10)]
    raw_train = pd.concat(train_chunks, ignore_index=True).sample(n=240000, random_state=SEED).reset_index(drop=True)

    raw_val = pd.concat(collected[10], ignore_index=True).sample(n=30000, random_state=SEED).reset_index(drop=True)

    test_chunks = [pd.concat(collected[m], ignore_index=True) for m in [11, 12]]
    raw_test = pd.concat(test_chunks, ignore_index=True).sample(n=30000, random_state=SEED).reset_index(drop=True)

    print(f"Raw splits collected: Train={len(raw_train):,}, Val={len(raw_val):,}, Test={len(raw_test):,}")

    # Compute historical baselines strictly from TRAINING SPLIT ONLY
    carrier_mean = raw_train.groupby('AIRLINE')['DEPARTURE_DELAY'].apply(lambda s: s.clip(lower=0).mean()).to_dict()
    airport_mean = raw_train.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].apply(lambda s: s.clip(lower=0).mean()).to_dict()
    hour_mean = raw_train.groupby(raw_train['SCHEDULED_DEPARTURE'] // 100)['DEPARTURE_DELAY'].apply(lambda s: s.clip(lower=0).mean()).to_dict()
    global_mean = float(raw_train['DEPARTURE_DELAY'].clip(lower=0).mean())

    print(f"Training split empirical delay mean: {global_mean:.2f} minutes.")

    def format_split(df_raw, split_name):
        dep_hour = (df_raw['SCHEDULED_DEPARTURE'] // 100).clip(0, 23).astype(int)
        dow = (df_raw['DAY_OF_WEEK'] - 1).clip(0, 6).astype(int)
        is_wknd = dow.isin([5, 6]).astype(int)
        is_pk = dep_hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        dist_km = (df_raw['DISTANCE'] * 1.60934).round(2)
        sched_dur = df_raw['SCHEDULED_TIME'].fillna(df_raw['DISTANCE'] / 8.0).clip(lower=15.0).round(1)

        # Baseline delay strictly from training rates:
        c_rate = df_raw['AIRLINE'].map(carrier_mean).fillna(global_mean)
        a_rate = df_raw['ORIGIN_AIRPORT'].map(airport_mean).fillna(global_mean)
        h_rate = dep_hour.map(hour_mean).fillna(global_mean)
        hist_delay = (0.4 * c_rate + 0.4 * a_rate + 0.2 * h_rate).round(2)

        # Join verified Open-Meteo weather
        wx_conditions = []
        temps = []
        origins = df_raw['ORIGIN_AIRPORT'].astype(str).values
        months = df_raw['MONTH'].astype(int).values
        days = df_raw['DAY'].astype(int).values

        for orig, m, d in zip(origins, months, days):
            day_key = f"{m:02d}-{d:02d}"
            wx_data = weather_db[orig]["days"][day_key]
            wx_conditions.append(wx_data['condition'])
            temps.append(float(wx_data['temperature_c']))

        delays = df_raw['DEPARTURE_DELAY'].clip(lower=0.0).astype(float).round(1)

        # Severity categories: 0 <= 5m, 1: 5-20m, 2: > 20m
        cat = np.where(delays <= 5.0, 0, np.where(delays <= 20.0, 1, 2))
        is_gt15 = (delays > 15.0).astype(int)

        formatted = pd.DataFrame({
            'transport_mode': 'AIRPLANE',
            'hour': dep_hour,
            'day_of_week': dow,
            'month': df_raw['MONTH'].astype(int),
            'day': df_raw['DAY'].astype(int),
            'is_weekend': is_wknd,
            'is_peak_hour': is_pk,
            'distance_km': dist_km,
            'scheduled_duration_min': sched_dur,
            'carrier_or_route': df_raw['AIRLINE'].astype(str),
            'origin': origins,
            'destination': df_raw['DESTINATION_AIRPORT'].astype(str),
            'city': origins, # airport code serves as hub identifier
            'historical_delay_mean': hist_delay,
            'vehicle_capacity': 180.0,  # Unused by model; kept for multimodal schema parity
            'occupancy_rate': 0.82,     # Unused by model; kept for multimodal schema parity
            'passenger_count': 148.0,   # Unused by model; kept for multimodal schema parity
            'weather_condition': wx_conditions,
            'temperature_c': temps,
            'headway_min': 0.0,         # N/A for flight departures
            'delay_minutes': delays,
            'delay_category': cat,
            'delay_gt15': is_gt15
        })

        out_path = os.path.join(PROCESSED_DIR, f"airplane_{split_name}.csv")
        formatted.to_csv(out_path, index=False)
        print(f"Saved {split_name} ({len(formatted):,} rows) to {out_path}.")

        # Print distribution
        on_time = (cat == 0).mean() * 100
        mod = (cat == 1).mean() * 100
        sev = (cat == 2).mean() * 100
        gt15 = is_gt15.mean() * 100
        print(f"  {split_name} Dist: On Time={on_time:.2f}%, Moderate={mod:.2f}%, Severe={sev:.2f}%, Delay>15m={gt15:.2f}%, Mean Delay={delays.mean():.2f}m")

    format_split(raw_train, "train")
    format_split(raw_val, "val")
    format_split(raw_test, "test")

    print(f"Extraction completed in {time.time() - start_t:.1f}s.")

if __name__ == "__main__":
    extract_and_process_flights()
