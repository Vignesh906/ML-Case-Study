"""
Synthetic Urban Transit Simulation Benchmark v2 Generator
-----------------------------------------------------------
Preserves the exact schema, column order (88 columns), column count, rows, trip_ids,
timestamps, routes, and stops from original transit/ directory.
Regenerates ONLY the delay-related values using an academically defensible,
mode-specific physics simulation model:
  Delay = corridor baseline + peak hour + occupancy/dwell + headway +
          weather severity (rain < snow < storm) + temperature extremes +
          mode physics (Metro grade-separated vs Bus in traffic) +
          zero-inflated right-skewed Gamma noise.
Recomputes route historical delay baselines from the training split only (leakage-free).
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
SRC_DIR = os.path.join(BASE_DIR, "transit")
DST_DIR = os.path.join(BASE_DIR, "transit_v2")

SEED = 42

def simulate_transit_delays(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Computes realistic synthetic delay values based on operational factors and physics.
    Does NOT modify input metadata, route identifiers, stops, or timestamps.
    """
    data = df.copy()
    n = len(data)

    # 1. Normalize mode
    raw_mode = data['route_type'].astype(str).str.lower()
    is_bus = (raw_mode == 'bus')
    is_rail = (raw_mode == 'rail')
    is_metro = (raw_mode == 'metro')

    # 2. Corridor length & scheduled duration baseline:
    length_km = pd.to_numeric(data['route_length_km'], errors='coerce').fillna(25.0).values
    duration_min = pd.to_numeric(data['scheduled_travel_min'], errors='coerce').fillna(35.0).values
    hour = pd.to_numeric(data['hour'], errors='coerce').fillna(12).values
    is_peak = pd.to_numeric(data['is_peak_hour'], errors='coerce').fillna(0).values
    occ = pd.to_numeric(data['occupancy_rate'], errors='coerce').fillna(0.65).values
    hw = pd.to_numeric(data['headway_min'], errors='coerce').fillna(10.0).values
    weather = data['weather_condition'].astype(str).str.lower().values
    temp = pd.to_numeric(data['temperature_c'], errors='coerce').fillna(18.0).values

    # Corridor baseline:
    corridor_bus = 0.5 + 0.025 * length_km
    corridor_rail = 0.4 + 0.010 * length_km + 0.012 * duration_min + np.where(duration_min > 60.0, (duration_min - 60.0) * 0.025, 0.0)
    corridor_metro = 0.4 + 0.010 * length_km
    corridor_base = np.where(is_bus, corridor_bus, np.where(is_rail, corridor_rail, corridor_metro))
    corridor_base = np.clip(corridor_base, 0.2, 7.0)

    # 3. Peak Hour Congestion (Morning 7-9, Evening 16-19)
    is_morning = ((hour >= 7) & (hour <= 9)).astype(float)
    is_evening = ((hour >= 16) & (hour <= 19)).astype(float)
    peak_bus = 3.5 * is_morning + 4.0 * is_evening
    peak_rail = 1.6 * is_peak
    peak_metro = 2.2 * is_morning + 2.5 * is_evening
    peak_penalty = np.where(is_bus, peak_bus, np.where(is_rail, peak_rail, peak_metro))

    # 4. Passenger Crowding & Dwell Time
    dwell_bus = np.maximum(0.0, (occ - 0.45) * 5.5) + np.where(occ > 0.85, 3.0, 0.0)
    dwell_rail = np.maximum(0.0, (occ - 0.50) * 3.8) + np.where(occ > 0.80, 2.0, 0.0)
    dwell_metro = np.maximum(0.0, (occ - 0.50) * 4.5) + np.where(occ > 0.80, 3.0, 0.0)
    dwell_penalty = np.where(is_bus, dwell_bus, np.where(is_rail, dwell_rail, dwell_metro))

    # 5. Service Headway Spacing
    hw_bus = np.maximum(0.0, (5.0 - hw) * 0.6) + np.maximum(0.0, (hw - 15.0) * 0.20)
    hw_rail = np.maximum(0.0, (hw - 30.0) * 0.12)
    hw_metro = np.maximum(0.0, (hw - 4.0) * 0.65) + np.where(hw > 8.0, 2.5, 0.0)
    headway_penalty = np.where(is_bus, hw_bus, np.where(is_rail, hw_rail, hw_metro))

    # 6. Meteorological Severity Hierarchy:
    wx_bus_map = {'clear': 0.0, 'cloudy': 0.5, 'fog': 1.6, 'light_rain': 2.5, 'heavy_rain': 5.5, 'snow': 8.0, 'storm': 12.0}
    wx_rail_map = {'clear': 0.0, 'cloudy': 0.4, 'fog': 1.5, 'light_rain': 2.4, 'heavy_rain': 5.5, 'snow': 8.0, 'storm': 12.0}
    # Metro: grade-separated underground; minimal weather sensitivity by design
    wx_metro_map = {'clear': 0.0, 'cloudy': 0.0, 'fog': 0.0, 'light_rain': 0.2, 'heavy_rain': 0.5, 'snow': 0.6, 'storm': 1.2}

    wx_bus = np.array([wx_bus_map.get(w, 0.0) for w in weather])
    wx_rail = np.array([wx_rail_map.get(w, 0.0) for w in weather])
    wx_metro = np.array([wx_metro_map.get(w, 0.0) for w in weather])
    wx_penalty = np.where(is_bus, wx_bus, np.where(is_rail, wx_rail, wx_metro))

    # 7. Ambient Temperature Extremes
    cold_pen = np.where(
        is_bus, np.maximum(0.0, (0.0 - temp) * 0.16),
        np.where(is_rail, np.maximum(0.0, (0.0 - temp) * 0.18),
                 np.maximum(0.0, (0.0 - temp) * 0.03))
    )
    heat_pen = np.where(
        is_bus, np.maximum(0.0, (temp - 32.0) * 0.16),
        np.where(is_rail, np.maximum(0.0, (temp - 32.0) * 0.18),
                 np.maximum(0.0, (temp - 35.0) * 0.05))
    )
    temp_penalty = cold_pen + heat_pen

    # 8. Interactions
    peak_occ_interaction = is_peak * np.where(occ > 0.70, 2.0, 0.0)
    storm_peak_interaction = (weather == 'storm').astype(float) * is_peak * 3.0
    interactions = peak_occ_interaction + storm_peak_interaction

    deterministic_delay = corridor_base + peak_penalty + dwell_penalty + headway_penalty + wx_penalty + temp_penalty + interactions

    # 9. Natural Dwell & Skewed Dispersion Formulation
    base_p_zero = np.where(is_bus, 0.42, np.where(is_rail, 0.55, 0.48))
    p_zero = np.clip(base_p_zero - 0.035 * deterministic_delay, 0.08, 0.72)
    is_delayed = rng.uniform(0.0, 1.0, size=n) > p_zero

    noise_sigma = np.where(is_bus, 1.8, np.where(is_rail, 1.8, 1.0))
    stochastic_noise = rng.normal(loc=0.0, scale=noise_sigma, size=n)

    # Operational shocks (accidents, switch failures, door interlocks)
    shock_p = np.where(is_bus, 0.052, np.where(is_rail, 0.048, 0.018))
    has_shock = rng.uniform(0.0, 1.0, size=n) < shock_p
    shock_delay = np.where(has_shock, 20.0 + rng.exponential(scale=6.5, size=n), 0.0)

    positive_delay = deterministic_delay + stochastic_noise + shock_delay

    # Final generated delay minutes (clamped and rounded to integer minutes)
    final_delays = np.where(is_delayed | has_shock, np.round(np.clip(positive_delay, 1.0, 85.0)), 0.0).astype(int)

    data['delay_minutes'] = final_delays

    # Recompute delay-derived columns consistently
    sched_min = pd.to_numeric(data['scheduled_travel_min'], errors='coerce').fillna(30.0).values
    data['actual_travel_min'] = np.round(sched_min + final_delays, 1)

    # on_time_flag: 1 if delay <= 5 mins else 0
    data['on_time_flag'] = (final_delays <= 5).astype(int)

    # service_status: 'on_time' (<=5m), 'minor_delay' (6-20m), 'major_delay' (>20m)
    status_col = np.where(
        final_delays <= 5, 'on_time',
        np.where(final_delays <= 20, 'minor_delay', 'major_delay')
    )
    data['service_status'] = status_col

    # reliability_score & consistency_score
    data['reliability_score'] = np.round(np.clip(1.0 - (final_delays / 25.0), 0.1, 1.0), 2)
    data['consistency_score'] = np.round(np.clip(1.0 - (final_delays / 35.0), 0.1, 1.0), 2)

    return data

def recompute_historical_delay_means(train_df, val_df, test_df):
    """
    Computes corridor historical baseline from TRAINING SPLIT ONLY.
    Maps back to all splits, preventing any temporal leakage.
    """
    print("Computing corridor historical delay mean from train split only...")
    route_means = train_df.groupby('route_id')['delay_minutes'].mean().to_dict()
    city_means = train_df.groupby('city')['delay_minutes'].mean().to_dict()
    overall_mean = train_df['delay_minutes'].mean()

    for df in [train_df, val_df, test_df]:
        hist = df['route_id'].map(route_means)
        hist = hist.fillna(df['city'].map(city_means)).fillna(overall_mean)
        df['delay_24h_mean'] = hist.round(2)
        df['delay_6h_mean'] = hist.round(2)
        df['delay_7d_mean'] = hist.round(2)
        df['delay_cumsum_4h'] = (hist * 2.5).round(2)

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)

    files_to_process = ['train.csv', 'val.csv', 'test (1).csv']
    dfs = {}

    for fname in files_to_process:
        src_path = os.path.join(SRC_DIR, fname)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing original source file: {src_path}")
        print(f"\nProcessing {fname} from {src_path}...")
        df_raw = pd.read_csv(src_path, low_memory=False)
        print(f"  Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

        # Simulate delays
        df_sim = simulate_transit_delays(df_raw, rng)
        dfs[fname] = df_sim

    # Recompute historical delay means without leakage from train split only
    recompute_historical_delay_means(dfs['train.csv'], dfs['val.csv'], dfs['test (1).csv'])

    # Verify schema preservation and save
    for fname in files_to_process:
        df_sim = dfs[fname]
        dst_path = os.path.join(DST_DIR, fname)
        src_path = os.path.join(SRC_DIR, fname)

        # Check column count and names
        src_cols = list(pd.read_csv(src_path, nrows=2).columns)
        assert list(df_sim.columns) == src_cols, f"Column mismatch in {fname}"
        assert len(df_sim) == len(pd.read_csv(src_path, usecols=['trip_id'])), f"Row count mismatch in {fname}"

        print(f"Saving {fname} to {dst_path}...")
        df_sim.to_csv(dst_path, index=False)
        print(f"  Successfully saved {fname}.")

    # Copy any auxiliary files from transit/ to transit_v2/
    for fname in os.listdir(SRC_DIR):
        if fname not in files_to_process and fname.endswith('.csv'):
            src_f = os.path.join(SRC_DIR, fname)
            dst_f = os.path.join(DST_DIR, fname)
            if not os.path.exists(dst_f):
                import shutil
                shutil.copyfile(src_f, dst_f)
                print(f"Copied auxiliary benchmark file {fname} to {DST_DIR}")

    print("\n[SUCCESS] transit_v2 generated successfully with verified identical schema!")

if __name__ == "__main__":
    main()
