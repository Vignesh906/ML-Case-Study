"""
Feature Engineering Module
--------------------------
Creates temporal, cyclical, operational, and interaction features
for multimodal transportation delay classification without data leakage.
"""

import numpy as np
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments input DataFrame with leakage-free engineered features.
    Defensively handles missing columns with operational defaults.
    """
    data = df.copy()

    # Defensive column defaults
    if 'hour' not in data.columns:
        data['hour'] = 12
    if 'day_of_week' not in data.columns:
        data['day_of_week'] = 2
    if 'month' not in data.columns:
        data['month'] = 6
    if 'day' not in data.columns:
        data['day'] = 15
    if 'transport_mode' not in data.columns:
        data['transport_mode'] = 'BUS'
    if 'weather_condition' not in data.columns:
        data['weather_condition'] = 'clear'

    # 1. Cyclical Temporal Representations
    hour = data['hour'].astype(float)
    dow = data['day_of_week'].astype(float)
    month = data['month'].astype(float)

    data['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    data['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    data['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    data['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    # 2. Temporal Dayparts
    data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 5)).astype(int)
    data['is_morning_rush'] = ((data['hour'] >= 7) & (data['hour'] <= 9)).astype(int)
    data['is_evening_rush'] = ((data['hour'] >= 16) & (data['hour'] <= 19)).astype(int)

    if 'is_peak_hour' not in data.columns:
        data['is_peak_hour'] = ((data['is_morning_rush'] == 1) | (data['is_evening_rush'] == 1)).astype(int)
    if 'is_weekend' not in data.columns:
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # 3. Operational & Interaction Features
    dist_val = data['distance_km'].astype(float) if 'distance_km' in data.columns else 25.0
    dist = np.maximum(dist_val, 0.1)
    dur_val = data['scheduled_duration_min'].astype(float) if 'scheduled_duration_min' in data.columns else 45.0
    duration = np.maximum(dur_val, 1.0)
    data['speed_kmh'] = (dist / (duration / 60.0)).clip(1.0, 950.0)

    cap_val = data['vehicle_capacity'].astype(float) if 'vehicle_capacity' in data.columns else 80.0
    cap = np.maximum(cap_val, 1.0)

    if 'occupancy_rate' in data.columns:
        occ_val = data['occupancy_rate'].astype(float).clip(0.0, 1.0)
    else:
        occ_val = pd.Series(0.65, index=data.index)

    if 'passenger_count' in data.columns:
        pax = np.maximum(data['passenger_count'].astype(float), 0.0)
    else:
        pax = np.maximum(cap * occ_val, 0.0)

    data['crowding_ratio'] = (pax / cap).clip(0.0, 2.5)

    # 4. Log-scaled dimensions for heavy-tailed distance/duration
    data['distance_log'] = np.log1p(dist)
    data['duration_log'] = np.log1p(duration)

    # 5. Modality Indicators and Operational Interactions
    data['is_mode_airplane'] = (data['transport_mode'] == 'AIRPLANE').astype(int)
    data['is_mode_bus'] = (data['transport_mode'] == 'BUS').astype(int)
    data['is_mode_rail'] = (data['transport_mode'] == 'TRAIN').astype(int)
    data['is_mode_metro'] = (data['transport_mode'] == 'METRO').astype(int)

    data['is_congested_period'] = ((data['is_peak_hour'] == 1) & (data['is_weekend'] == 0)).astype(int)
    data['is_adverse_weather'] = data['weather_condition'].isin(['heavy_rain', 'storm', 'snow', 'fog']).astype(int)
    data['is_severe_storm'] = (data['weather_condition'] == 'storm').astype(int)
    data['weather_congestion_interaction'] = (data['is_adverse_weather'] * data['is_peak_hour']).astype(int)
    hist_val = data['historical_delay_mean'].astype(float) if 'historical_delay_mean' in data.columns else 1.5
    hist_delay = np.maximum(hist_val, 0.0)
    peak = data['is_peak_hour'].astype(float)
    wknd = data['is_weekend'].astype(float)

    # Operational Delay Risk Index incorporating historical baseline, peak congestion, crowding, and meteorology
    adv_w = data['is_adverse_weather'].astype(float)
    storm_w = data['is_severe_storm'].astype(float)
    crowd_factor = (data['crowding_ratio'] > 0.85).astype(float)
    data['delay_risk_score'] = (
        hist_delay * (
            1.0 + 
            0.35 * peak + 
            0.15 * wknd + 
            0.30 * adv_w + 
            0.25 * storm_w + 
            0.20 * crowd_factor
        )
    ).clip(0.0, 250.0)

    data['aviation_long_haul'] = ((data['transport_mode'] == 'AIRPLANE') & (dist > 1500.0)).astype(int)
    hw = data['headway_min'].astype(float) if 'headway_min' in data.columns else pd.Series(0.0, index=data.index)
    data['transit_high_headway'] = ((data['transport_mode'] != 'AIRPLANE') & (hw > 12.0)).astype(int)

    return data

if __name__ == "__main__":
    test_df = pd.DataFrame({
        'transport_mode': ['BUS', 'AIRPLANE'],
        'hour': [8, 17],
        'day_of_week': [1, 4],
        'month': [3, 11],
        'is_peak_hour': [1, 1],
        'is_weekend': [0, 0],
        'distance_km': [12.5, 1200.0],
        'scheduled_duration_min': [45.0, 160.0],
        'vehicle_capacity': [80.0, 180.0],
        'passenger_count': [65.0, 150.0],
        'historical_delay_mean': [2.5, 22.0]
    })
    res = add_engineered_features(test_df)
    print("Engineered columns:", list(res.columns))
    print(res[['speed_kmh', 'crowding_ratio', 'delay_risk_score']])
