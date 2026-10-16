import os
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TRANSIT_DIR = os.path.join(BASE_DIR, "transit_v2")

print("Generating unified multimodal datasets from transit_v2 and airplane processed splits...")

# 1. Load transit splits
train_tr = pd.read_csv(os.path.join(TRANSIT_DIR, "train.csv"), low_memory=False)
val_tr = pd.read_csv(os.path.join(TRANSIT_DIR, "val.csv"), low_memory=False)
test_tr = pd.read_csv(os.path.join(TRANSIT_DIR, "test (1).csv"), low_memory=False)

def format_transit(df, samples_per_mode=25000, seed=42):
    mode_map = {'bus': 'BUS', 'rail': 'TRAIN', 'metro': 'METRO'}
    parts = []
    for rtype, tmode in mode_map.items():
        sub = df[df['route_type'].astype(str).str.lower() == rtype].copy()
        if len(sub) > samples_per_mode:
            sub = sub.sample(n=samples_per_mode, random_state=seed)
        
        delays = sub['delay_minutes'].clip(lower=0).astype(float)
        cat = sub['service_status'].map({'on_time': 0, 'minor_delay': 1, 'major_delay': 2}).fillna(0).astype(int)
        
        parsed_dt = pd.to_datetime(sub['timestamp'], errors='coerce')
        formatted = pd.DataFrame({
            'transport_mode': tmode,
            'hour': sub['hour'].astype(int),
            'day_of_week': sub['day_of_week'].astype(int),
            'month': sub['month'].astype(int),
            'day': parsed_dt.dt.day.fillna(15).astype(int),
            'is_weekend': sub['is_weekend'].astype(int),
            'is_peak_hour': sub['is_peak_hour'].astype(int),
            'distance_km': sub['route_length_km'].astype(float),
            'scheduled_duration_min': sub['scheduled_travel_min'].astype(float),
            'carrier_or_route': sub['route_id'].astype(str),
            'origin': sub['origin_station'].astype(str),
            'destination': sub['dest_station'].astype(str),
            'city': sub['city'].astype(str),
            'historical_delay_mean': sub['delay_24h_mean'].fillna(1.5).astype(float),
            'vehicle_capacity': sub['vehicle_capacity'].astype(float),
            'occupancy_rate': sub['occupancy_rate'].astype(float),
            'passenger_count': sub['passenger_count'].astype(float),
            'weather_condition': sub['weather_condition'].astype(str),
            'temperature_c': sub['temperature_c'].astype(float),
            'headway_min': sub['headway_min'].astype(float),
            'delay_minutes': delays,
            'delay_category': cat
        })
        parts.append(formatted)
    return pd.concat(parts, ignore_index=True)

# 2. Format transit splits
tr_trans_df = format_transit(train_tr, samples_per_mode=25000, seed=42)
vl_trans_df = format_transit(val_tr, samples_per_mode=5000, seed=42)
ts_trans_df = format_transit(test_tr, samples_per_mode=5000, seed=42)

# 3. Load airplane splits
tr_air = pd.read_csv(os.path.join(PROCESSED_DIR, "airplane_train.csv"), low_memory=False)
vl_air = pd.read_csv(os.path.join(PROCESSED_DIR, "airplane_val.csv"), low_memory=False)
ts_air = pd.read_csv(os.path.join(PROCESSED_DIR, "airplane_test.csv"), low_memory=False)

# Sample airplane for pooled multimodal sets
air_cols = [c for c in tr_trans_df.columns]
tr_air_sub = tr_air[air_cols].sample(n=25000, random_state=42)
vl_air_sub = vl_air[air_cols].sample(n=5000, random_state=42)
ts_air_sub = ts_air[air_cols].sample(n=5000, random_state=42)

# Concatenate and shuffle
mm_train = pd.concat([tr_trans_df, tr_air_sub], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
mm_val = pd.concat([vl_trans_df, vl_air_sub], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
mm_test = pd.concat([ts_trans_df, ts_air_sub], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

mm_train.to_csv(os.path.join(PROCESSED_DIR, "multimodal_train.csv"), index=False)
mm_val.to_csv(os.path.join(PROCESSED_DIR, "multimodal_val.csv"), index=False)
mm_test.to_csv(os.path.join(PROCESSED_DIR, "multimodal_test.csv"), index=False)

print(f"Generated multimodal_train.csv: {len(mm_train):,} rows")
print(f"Generated multimodal_val.csv: {len(mm_val):,} rows")
print(f"Generated multimodal_test.csv: {len(mm_test):,} rows")
