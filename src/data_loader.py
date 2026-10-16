"""
Multimodal Data Loader and Harmonization Pipeline
--------------------------------------------------
Extracts, cleans, standardizes, and unifies public transit (Bus, Train/Rail, Metro)
from Synthetic Urban Transit Simulation Benchmark v2 (transit_v2) and commercial aviation
(Airplane) records from authentic US Department of Transportation BTS 2015 records.
Features genuine historical Open-Meteo weather matching airport coordinates and dates.
Zero target leakage: strictly excludes post-flight attributes (ARRIVAL_DELAY, TAXI_IN,
WHEELS_ON, WEATHER_DELAY cause attributions). Carrier and airport historical baselines
are computed strictly from training data only.
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TRANSIT_DIR = os.path.join(BASE_DIR, "transit_v2")
RAW_ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
AIRPORT_WEATHER_FILE = os.path.join(BASE_DIR, "data", "historical_airport_weather_2015.json")

def get_delay_category(delay_val):
    """
    Project-defined severity categories:
      0 = ON TIME (<= 5 min)
      1 = MODERATELY DELAYED (> 5 to <= 20 min)
      2 = SEVERELY DELAYED (> 20 min)
    """
    if pd.isna(delay_val):
        return 0
    d = float(delay_val)
    if d <= 5.0:
        return 0
    elif d <= 20.0:
        return 1
    else:
        return 2

def load_airport_metadata():
    """Load airports.csv to map IATA airport codes to cities."""
    airports_path = os.path.join(RAW_ARCHIVE_DIR, "airports.csv")
    if os.path.exists(airports_path):
        df_airports = pd.read_csv(airports_path)
        return dict(zip(df_airports['IATA_CODE'], df_airports['CITY']))
    return {}

def load_historical_airport_weather():
    """Loads genuine 2015 historical weather observations from Open-Meteo archive."""
    if os.path.exists(AIRPORT_WEATHER_FILE):
        try:
            with open(AIRPORT_WEATHER_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {AIRPORT_WEATHER_FILE}: {e}")
    return {}

def extract_transit_partition(file_name, samples_per_mode=20000, seed=42):
    """
    Extracts and standardizes transit records for BUS, TRAIN (rail), and METRO
    from the validated transit_v2 benchmark.
    """
    file_path = os.path.join(RAW_TRANSIT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transit file not found: {file_path}")

    cols = [
        'timestamp', 'route_id', 'city', 'route_type',
        'origin_station', 'dest_station', 'passenger_count',
        'occupancy_rate', 'delay_minutes', 'scheduled_travel_min',
        'weather_condition', 'temperature_c', 'route_length_km', 'headway_min',
        'vehicle_capacity', 'is_peak_hour', 'is_weekend', 'hour', 'day_of_week',
        'month', 'delay_24h_mean'
    ]
    df = pd.read_csv(file_path, usecols=cols)
    mode_map = {'bus': 'BUS', 'rail': 'TRAIN', 'metro': 'METRO'}
    partitions = []

    for rtype, tmode in mode_map.items():
        sub = df[df['route_type'] == rtype].copy()
        if len(sub) > samples_per_mode:
            sub = sub.sample(n=samples_per_mode, random_state=seed)

        parsed_dt = pd.to_datetime(sub['timestamp'], errors='coerce')
        sub_mapped = pd.DataFrame({
            'transport_mode': tmode,
            'hour': sub['hour'].astype(int),
            'day_of_week': sub['day_of_week'].astype(int),
            'month': sub['month'].astype(int),
            'day': parsed_dt.dt.day.fillna(1).astype(int),
            'is_weekend': sub['is_weekend'].astype(int),
            'is_peak_hour': sub['is_peak_hour'].astype(int),
            'distance_km': sub['route_length_km'].astype(float),
            'scheduled_duration_min': sub['scheduled_travel_min'].astype(float),
            'carrier_or_route': sub['route_id'].astype(str),
            'origin': sub['origin_station'].astype(str),
            'destination': sub['dest_station'].astype(str),
            'city': sub['city'].astype(str),
            'historical_delay_mean': sub['delay_24h_mean'].fillna(0.0).astype(float),
            'vehicle_capacity': sub['vehicle_capacity'].astype(float),
            'occupancy_rate': sub['occupancy_rate'].astype(float),
            'passenger_count': sub['passenger_count'].astype(float),
            'weather_condition': sub['weather_condition'].astype(str),
            'temperature_c': sub['temperature_c'].astype(float),
            'headway_min': sub['headway_min'].astype(float),
            'delay_minutes': sub['delay_minutes'].clip(lower=0).astype(float)
        })
        sub_mapped['delay_category'] = sub_mapped['delay_minutes'].apply(get_delay_category)
        partitions.append(sub_mapped)

    return pd.concat(partitions, ignore_index=True)

def extract_flight_partitions(train_target=20000, val_target=3000, test_target=3000, seed=42):
    """
    Streams raw flights.csv and extracts balanced chronological subsets for AIRPLANE mode.
    Pre-flight inputs only: NO post-flight attributes (ARRIVAL_DELAY, TAXI_IN, WHEELS_ON,
    WEATHER_DELAY). Joins genuine 2015 historical weather from Open-Meteo archive.
    Carrier and airport historical delay baselines are computed from training data only.
    """
    flights_path = os.path.join(RAW_ARCHIVE_DIR, "flights.csv")
    if not os.path.exists(flights_path):
        raise FileNotFoundError(f"Flights file not found: {flights_path}")

    airport_city_map = load_airport_metadata()
    weather_db = load_historical_airport_weather()

    # Pre-flight columns strictly: NO post-flight or arrival leak features
    cols = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'DISTANCE',
        'CANCELLED', 'DIVERTED'
    ]

    # Stratified target per month to avoid seasonal shift
    # Train: months 1..9 (approx 2,222 per month)
    # Val: month 10 (3,000)
    # Test: months 11..12 (1,500 per month)
    per_month_train_target = int(np.ceil(train_target / 9.0))
    per_month_test_target = int(np.ceil(test_target / 2.0))

    month_chunks = {m: [] for m in range(1, 13)}

    for chunk in pd.read_csv(flights_path, usecols=cols, chunksize=500000, low_memory=False):
        chunk['DEPARTURE_DELAY'] = pd.to_numeric(chunk['DEPARTURE_DELAY'], errors='coerce')
        chunk['SCHEDULED_TIME'] = pd.to_numeric(chunk['SCHEDULED_TIME'], errors='coerce')
        chunk['DISTANCE'] = pd.to_numeric(chunk['DISTANCE'], errors='coerce')
        chunk['SCHEDULED_DEPARTURE'] = pd.to_numeric(chunk['SCHEDULED_DEPARTURE'], errors='coerce').fillna(800)
        chunk['CANCELLED'] = pd.to_numeric(chunk['CANCELLED'], errors='coerce').fillna(0)
        chunk['DIVERTED'] = pd.to_numeric(chunk['DIVERTED'], errors='coerce').fillna(0)

        # Exclude cancellations/diversions and missing delay values
        valid = chunk[(chunk['CANCELLED'] == 0) & (chunk['DIVERTED'] == 0) & (chunk['DEPARTURE_DELAY'].notna())]

        for m_val in range(1, 13):
            sub_m = valid[valid['MONTH'] == m_val]
            if len(sub_m) == 0:
                continue

            current_cnt = sum(len(x) for x in month_chunks[m_val])
            if m_val <= 9:
                target_cnt = per_month_train_target
            elif m_val == 10:
                target_cnt = val_target
            else:
                target_cnt = per_month_test_target

            if current_cnt < target_cnt:
                take = min(target_cnt - current_cnt, len(sub_m))
                month_chunks[m_val].append(sub_m.sample(n=take, random_state=seed))

        # Check if all targets met
        all_train_done = all(sum(len(x) for x in month_chunks[m]) >= per_month_train_target for m in range(1, 10))
        val_done = sum(len(x) for x in month_chunks[10]) >= val_target
        test_done = all(sum(len(x) for x in month_chunks[m]) >= per_month_test_target for m in [11, 12])

        if all_train_done and val_done and test_done:
            break

    # Assemble raw splits
    raw_train_fl = pd.concat([pd.concat(month_chunks[m], ignore_index=True) for m in range(1, 10)], ignore_index=True)
    if len(raw_train_fl) > train_target:
        raw_train_fl = raw_train_fl.sample(n=train_target, random_state=seed).reset_index(drop=True)

    raw_val_fl = pd.concat(month_chunks[10], ignore_index=True)
    if len(raw_val_fl) > val_target:
        raw_val_fl = raw_val_fl.sample(n=val_target, random_state=seed).reset_index(drop=True)

    raw_test_fl = pd.concat([pd.concat(month_chunks[m], ignore_index=True) for m in [11, 12]], ignore_index=True)
    if len(raw_test_fl) > test_target:
        raw_test_fl = raw_test_fl.sample(n=test_target, random_state=seed).reset_index(drop=True)

    # Compute carrier and airport historical delay mean from TRAINING SPLIT ONLY (Leakage-Free)
    carrier_hist_delay_map = raw_train_fl.groupby('AIRLINE')['DEPARTURE_DELAY'].apply(lambda s: s.clip(lower=0).mean()).to_dict()
    airport_hist_delay_map = raw_train_fl.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].apply(lambda s: s.clip(lower=0).mean()).to_dict()
    flight_overall_mean = float(raw_train_fl['DEPARTURE_DELAY'].clip(lower=0).mean())

    seasonal_temp_map = {
        1: 2.0, 2: 4.0, 3: 10.0, 4: 15.0, 5: 20.0, 6: 25.0,
        7: 28.0, 8: 27.0, 9: 22.0, 10: 15.0, 11: 9.0, 12: 3.5
    }

    def format_flights(raw_df):
        dep_hour = (raw_df['SCHEDULED_DEPARTURE'] // 100).clip(0, 23).astype(int)
        dow = (raw_df['DAY_OF_WEEK'] - 1).clip(0, 6).astype(int)
        is_wknd = dow.isin([5, 6]).astype(int)
        is_pk = dep_hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        dist_km = (raw_df['DISTANCE'] * 1.60934).round(2)
        sched_dur = raw_df['SCHEDULED_TIME'].fillna(raw_df['DISTANCE'] / 8.0).clip(lower=15.0).round(1)
        origin_str = raw_df['ORIGIN_AIRPORT'].astype(str)
        dest_str = raw_df['DESTINATION_AIRPORT'].astype(str)
        city_str = origin_str.map(airport_city_map).fillna("Airport City")

        # Historical delay baseline: combined carrier and origin airport empirical baseline
        carrier_h = raw_df['AIRLINE'].map(carrier_hist_delay_map).fillna(flight_overall_mean)
        airport_h = raw_df['ORIGIN_AIRPORT'].map(airport_hist_delay_map).fillna(flight_overall_mean)
        hist_delay = (0.5 * carrier_h + 0.5 * airport_h).round(2).astype(float)

        delay_min = raw_df['DEPARTURE_DELAY'].clip(lower=0.0).astype(float)

        # Match genuine historical weather from Open-Meteo archive
        wx_conditions = []
        temps = []
        months = raw_df['MONTH'].values
        days = raw_df['DAY'].values
        origins = origin_str.values

        for orig, m, d in zip(origins, months, days):
            day_key = f"{int(m):02d}-{int(d):02d}"
            airport_data = weather_db.get(orig, {}).get("days", {}).get(day_key)
            if airport_data:
                wx_conditions.append(airport_data.get('condition', 'clear'))
                temps.append(float(airport_data.get('temperature_c', seasonal_temp_map.get(m, 18.0))))
            else:
                wx_conditions.append('clear')
                temps.append(float(seasonal_temp_map.get(m, 18.0)))

        formatted = pd.DataFrame({
            'transport_mode': 'AIRPLANE',
            'hour': dep_hour,
            'day_of_week': dow,
            'month': raw_df['MONTH'].astype(int),
            'day': raw_df['DAY'].astype(int),
            'is_weekend': is_wknd,
            'is_peak_hour': is_pk,
            'distance_km': dist_km,
            'scheduled_duration_min': sched_dur,
            'carrier_or_route': raw_df['AIRLINE'].astype(str),
            'origin': origin_str,
            'destination': dest_str,
            'city': city_str,
            'historical_delay_mean': hist_delay,
            'vehicle_capacity': 180.0,
            'occupancy_rate': 0.82,
            'passenger_count': 148.0,
            'weather_condition': wx_conditions,
            'temperature_c': temps,
            'headway_min': 0.0,
            'delay_minutes': delay_min
        })
        formatted['delay_category'] = formatted['delay_minutes'].apply(get_delay_category)
        return formatted

    df_tr_fl = format_flights(raw_train_fl)
    df_vl_fl = format_flights(raw_val_fl)
    df_ts_fl = format_flights(raw_test_fl)

    return df_tr_fl, df_vl_fl, df_ts_fl

def build_multimodal_datasets(samples_per_mode_train=20000, samples_per_mode_val=3000, samples_per_mode_test=3000):
    """
    Constructs multimodal training, validation, and testing sets and saves them to data/processed/.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Building multimodal datasets from transit_v2 and authentic DOT aviation records...")

    # 1. Public Transit splits (from transit_v2)
    print("Extracting public transit training data from transit_v2/train.csv...")
    tr_transit = extract_transit_partition("train.csv", samples_per_mode=samples_per_mode_train, seed=42)
    print("Extracting public transit validation data from transit_v2/val.csv...")
    vl_transit = extract_transit_partition("val.csv", samples_per_mode=samples_per_mode_val, seed=42)
    print("Extracting public transit test data from transit_v2/test (1).csv...")
    ts_transit = extract_transit_partition("test (1).csv", samples_per_mode=samples_per_mode_test, seed=42)

    # 2. Flight splits (authentic DOT, leakage-free Open-Meteo weather)
    print("Extracting commercial flight data (leakage-free)...")
    tr_flight, vl_flight, ts_flight = extract_flight_partitions(
        train_target=samples_per_mode_train,
        val_target=samples_per_mode_val,
        test_target=samples_per_mode_test,
        seed=42
    )

    # 3. Concatenate and shuffle
    train_df = pd.concat([tr_transit, tr_flight], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = pd.concat([vl_transit, vl_flight], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = pd.concat([ts_transit, ts_flight], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 4. Save to disk
    train_path = os.path.join(PROCESSED_DIR, "multimodal_train.csv")
    val_path = os.path.join(PROCESSED_DIR, "multimodal_val.csv")
    test_path = os.path.join(PROCESSED_DIR, "multimodal_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved {len(train_df)} training records to {train_path}")
    print(f"Saved {len(val_df)} validation records to {val_path}")
    print(f"Saved {len(test_df)} test records to {test_path}")

    return train_df, val_df, test_df

# Alias for backwards compatibility
build_multimodal_splits = build_multimodal_datasets

def load_processed_data():
    """Loads the pre-built multimodal train, validation, and test sets."""
    train_path = os.path.join(PROCESSED_DIR, "multimodal_train.csv")
    val_path = os.path.join(PROCESSED_DIR, "multimodal_val.csv")
    test_path = os.path.join(PROCESSED_DIR, "multimodal_test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        return build_multimodal_datasets()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

if __name__ == "__main__":
    build_multimodal_datasets()
