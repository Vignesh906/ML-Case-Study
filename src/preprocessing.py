"""
Data Preprocessing Pipeline
---------------------------
Constructs, fits, saves, and applies a Scikit-Learn ColumnTransformer pipeline
with Standard Scaler for numerical features and OneHotEncoder for categorical features.
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from src.feature_engineering import add_engineered_features

CATEGORICAL_FEATURES = ['transport_mode', 'weather_condition']

NUMERICAL_FEATURES = [
    'hour', 'day_of_week', 'month', 'day', 'is_weekend', 'is_peak_hour',
    'distance_km', 'scheduled_duration_min', 'historical_delay_mean',
    'vehicle_capacity', 'occupancy_rate', 'passenger_count', 'temperature_c',
    'headway_min', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin',
    'month_cos', 'is_night', 'is_morning_rush', 'is_evening_rush',
    'speed_kmh', 'crowding_ratio', 'delay_risk_score', 'distance_log', 'duration_log',
    'is_mode_airplane', 'is_mode_bus', 'is_mode_rail', 'is_mode_metro',
    'is_congested_period', 'is_adverse_weather', 'is_severe_storm',
    'weather_congestion_interaction', 'aviation_long_haul', 'transit_high_headway'
]

ALL_INPUT_COLUMNS = [
    'transport_mode', 'hour', 'day_of_week', 'month', 'day',
    'is_weekend', 'is_peak_hour', 'distance_km', 'scheduled_duration_min',
    'historical_delay_mean', 'vehicle_capacity', 'occupancy_rate',
    'passenger_count', 'weather_condition', 'temperature_c', 'headway_min'
]

def build_column_transformer():
    """Builds a scikit-learn ColumnTransformer for categorical and numerical features."""
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, CATEGORICAL_FEATURES),
            ('num', num_transformer, NUMERICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

class MultimodalPreprocessor:
    """End-to-end preprocessing wrapper including feature engineering and column transformations."""
    def __init__(self):
        self.column_transformer = build_column_transformer()
        self.is_fitted = False
        self.feature_names_out_ = None

    def fit(self, df: pd.DataFrame):
        df_feat = add_engineered_features(df)
        self.column_transformer.fit(df_feat)
        self.is_fitted = True
        self.feature_names_out_ = [
            f.replace('cat__', '').replace('num__', '')
            for f in self.column_transformer.get_feature_names_out()
        ]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        df_feat = add_engineered_features(df)
        return self.column_transformer.transform(df_feat)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self):
        return self.feature_names_out_

def save_preprocessor(preprocessor: MultimodalPreprocessor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")

def load_preprocessor(path: str) -> MultimodalPreprocessor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found at {path}")
    return joblib.load(path)

if __name__ == "__main__":
    from src.data_loader import load_processed_data
    train_df, _, _ = load_processed_data()
    prep = MultimodalPreprocessor()
    X = prep.fit_transform(train_df.head(100))
    print("Transformed shape:", X.shape)
    print("Feature count:", len(prep.get_feature_names()))
    print("Features preview:", prep.get_feature_names()[:10])
