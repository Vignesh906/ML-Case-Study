"""
Comprehensive Automated Test Suite
-----------------------------------
Tests:
1. Target categorization logic
2. Feature engineering transformations
3. Multimodal preprocessing pipeline
4. Model artifacts loading
5. End-to-end prediction engine
6. Robustness to missing/invalid inputs
7. Flask application routes and API responses
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import get_delay_category
from src.feature_engineering import add_engineered_features
from src.preprocessing import MultimodalPreprocessor, load_preprocessor
from src.prediction import DelayPredictor, predictor_instance

class TestMultimodalDelaySystem(unittest.TestCase):

    def test_target_creation(self):
        """Test delay category discretization (0: 0-5m, 1: 6-20m, 2: >20m)."""
        self.assertEqual(get_delay_category(0), 0)
        self.assertEqual(get_delay_category(5), 0)
        self.assertEqual(get_delay_category(5.5), 1)
        self.assertEqual(get_delay_category(6), 1)
        self.assertEqual(get_delay_category(20), 1)
        self.assertEqual(get_delay_category(20.1), 2)
        self.assertEqual(get_delay_category(65), 2)
        self.assertEqual(get_delay_category(np.nan), 0)

    def test_feature_engineering(self):
        """Test cyclical and operational feature additions."""
        sample_df = pd.DataFrame([{
            'transport_mode': 'BUS',
            'hour': 8,
            'day_of_week': 2,
            'month': 5,
            'day': 12,
            'is_weekend': 0,
            'is_peak_hour': 1,
            'distance_km': 15.0,
            'scheduled_duration_min': 30.0,
            'historical_delay_mean': 4.0,
            'vehicle_capacity': 80.0,
            'occupancy_rate': 0.75,
            'passenger_count': 60.0,
            'weather_condition': 'clear',
            'temperature_c': 20.0,
            'headway_min': 10.0
        }])
        feat_df = add_engineered_features(sample_df)
        self.assertIn('hour_sin', feat_df.columns)
        self.assertIn('hour_cos', feat_df.columns)
        self.assertIn('speed_kmh', feat_df.columns)
        self.assertIn('crowding_ratio', feat_df.columns)
        self.assertIn('delay_risk_score', feat_df.columns)
        self.assertAlmostEqual(feat_df['speed_kmh'].iloc[0], 30.0, places=1)
        self.assertAlmostEqual(feat_df['crowding_ratio'].iloc[0], 0.75, places=2)

    def test_preprocessor_loading_and_transform(self):
        """Test loading preprocessor pipeline and transforming arbitrary sample."""
        prep_path = os.path.join(BASE_DIR, "models", "preprocessing_pipeline.joblib")
        self.assertTrue(os.path.exists(prep_path), f"Preprocessor missing at {prep_path}")
        prep = load_preprocessor(prep_path)
        sample_df = pd.DataFrame([{
            'transport_mode': 'AIRPLANE',
            'hour': 14,
            'day_of_week': 4,
            'month': 8,
            'day': 20,
            'is_weekend': 0,
            'is_peak_hour': 0,
            'distance_km': 1200.0,
            'scheduled_duration_min': 150.0,
            'carrier_or_route': 'DL',
            'origin': 'ATL',
            'destination': 'LGA',
            'city': 'Atlanta',
            'historical_delay_mean': 12.0,
            'vehicle_capacity': 180.0,
            'occupancy_rate': 0.85,
            'passenger_count': 153.0,
            'weather_condition': 'cloudy',
            'temperature_c': 26.0,
            'headway_min': 0.0,
            'delay_minutes': 0.0
        }])
        X_vec = prep.transform(sample_df)
        self.assertEqual(len(X_vec.shape), 2)
        self.assertEqual(X_vec.shape[0], 1)
        self.assertGreater(X_vec.shape[1], 30)

    def test_prediction_engine_all_modes(self):
        """Test real prediction across all 4 modes."""
        predictor = DelayPredictor()
        self.assertTrue(predictor.is_ready())

        for mode in ['BUS', 'TRAIN', 'METRO', 'AIRPLANE']:
            res = predictor.predict({
                'transport_mode': mode,
                'hour': 10,
                'day_of_week': 2,
                'distance_km': 50.0 if mode != 'AIRPLANE' else 900.0,
                'scheduled_duration_min': 60.0 if mode != 'AIRPLANE' else 110.0,
                'historical_delay_mean': 3.0,
                'occupancy_rate': 0.6
            })
            self.assertEqual(res['status'], 'success')
            self.assertEqual(res['transport_mode'], mode)
            self.assertIn(res['predicted_category'], ['ON TIME', 'MODERATELY DELAYED', 'SEVERELY DELAYED'])
            self.assertIn('probabilities', res)
            self.assertAlmostEqual(sum(res['probabilities'].values()), 100.0, delta=1.0)

    def test_invalid_input_handling(self):
        """Test engine handles missing or extreme inputs gracefully without crashing."""
        predictor = DelayPredictor()
        res_empty = predictor.predict({})
        self.assertEqual(res_empty['status'], 'success')
        self.assertIn(res_empty['predicted_category'], ['ON TIME', 'MODERATELY DELAYED', 'SEVERELY DELAYED'])

        res_extreme = predictor.predict({
            'transport_mode': 'UNKNOWN_MODE',
            'hour': 999,
            'distance_km': -500.0,
            'temperature_c': 1000.0
        })
        self.assertEqual(res_extreme['status'], 'success')

    def test_flask_routes(self):
        """Test Flask web routes and API endpoints."""
        from app import app
        client = app.test_client()

        # Web pages
        for route in ['/', '/prediction', '/models', '/analytics', '/about']:
            res = client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed with status {res.status_code}")

        # Stats API
        stats_res = client.get('/api/stats')
        self.assertEqual(stats_res.status_code, 200)
        stats_json = stats_res.get_json()
        self.assertIn('dataset', stats_json)
        self.assertIn('metrics', stats_json)

        # Predict API
        pred_res = client.post('/api/predict', json={
            'transport_mode': 'BUS',
            'hour': 9,
            'day_of_week': 3,
            'distance_km': 20.0,
            'scheduled_duration_min': 50.0,
            'historical_delay_mean': 1.0
        })
        self.assertEqual(pred_res.status_code, 200)
        pred_json = pred_res.get_json()
        self.assertEqual(pred_json['status'], 'success')
        self.assertIn('predicted_category', pred_json)
        self.assertIn('probabilities', pred_json)

if __name__ == "__main__":
    unittest.main()
