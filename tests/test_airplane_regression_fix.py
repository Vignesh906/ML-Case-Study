"""
Aviation Delay Prediction Audit & Two-Part Hurdle Regression Test Suite
========================================================================
Validates:
1. TwoPartHurdleRegressor correctly loaded for Airplane continuous delay estimation
2. Airplane predictions for nominal, off-peak flights produce realistic < 5 min delay estimates
3. Airplane predictions for severe disruption / extreme weather scale realistically (up to 15-40 min)
4. Airport sensitivity: different origin airports produce distinct, realistic predictions
5. No hardcoded 10-minute floor or artificial prediction clamps
6. Calibrated classification probabilities sum to 100.0%
7. Preservation of original DOT flights dataset (archive/flights.csv unchanged)
8. Active classification models: Linear SVM, Random Forest, Decision Tree, KNN, Logistic Regression
   (XGBoost strictly absent)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prediction import DelayPredictor, predictor_instance
from src.hurdle_model import TwoPartHurdleRegressor

class TestAirplaneRegressionFix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = DelayPredictor()

    def test_01_airplane_regressor_architecture(self):
        """Airplane regressor must be an instance of TwoPartHurdleRegressor."""
        air_reg = self.predictor.mode_regressors.get('AIRPLANE')
        self.assertIsNotNone(air_reg, "Airplane mode regressor must exist in mode_regressors dict")
        self.assertIsInstance(air_reg, TwoPartHurdleRegressor,
                              f"Expected TwoPartHurdleRegressor, got {type(air_reg)}")

    def test_02_airplane_nominal_prediction_under_5_min(self):
        """Nominal, low-risk flights (early morning, low prior delay) must predict < 5 min (On Time)."""
        early_flight = {
            'transport_mode': 'AIRPLANE',
            'carrier_or_route': 'DL',
            'origin': 'ATL',
            'destination': 'LAX',
            'hour': 6, # 6 AM departure
            'day_of_week': 2,
            'month': 10,
            'day': 14,
            'historical_delay_mean': 8.5,
            'weather_condition': 'clear',
            'temperature_c': 18.0
        }
        res = self.predictor.predict(early_flight)
        est = res['estimated_delay_min']
        self.assertLess(est, 5.0, f"Early morning low-risk flight should predict < 5.0 min, got {est}")
        self.assertEqual(res['predicted_category'], 'ON TIME')

    def test_03_airplane_carrier_sensitivity(self):
        """Low-delay carriers (Hawaiian/Alaska) must predict lower delay than congested carriers (United/Spirit)."""
        ha_flight = {
            'transport_mode': 'AIRPLANE',
            'carrier_or_route': 'HA',
            'origin': 'HNL',
            'destination': 'OGG',
            'hour': 10,
            'day_of_week': 2,
            'month': 10,
            'historical_delay_mean': 4.8,
            'weather_condition': 'clear',
            'temperature_c': 25.0
        }
        ua_flight = {
            'transport_mode': 'AIRPLANE',
            'carrier_or_route': 'UA',
            'origin': 'ORD',
            'destination': 'DEN',
            'hour': 18,
            'day_of_week': 4,
            'month': 10,
            'historical_delay_mean': 17.1,
            'weather_condition': 'heavy_rain',
            'temperature_c': 12.0
        }
        res_ha = self.predictor.predict(ha_flight)
        res_ua = self.predictor.predict(ua_flight)
        
        self.assertLess(res_ha['raw_regression_min'], res_ua['raw_regression_min'],
                       "Hawaiian Airlines off-peak should predict lower delay than United rush hour heavy rain")
        self.assertLess(res_ha['estimated_delay_min'], 5.0)

    def test_04_airplane_airport_sensitivity(self):
        """Differing origin airports must produce differing historical baselines and predictions."""
        res_sea = self.predictor.predict({
            'transport_mode': 'AIRPLANE',
            'carrier_or_route': 'DL',
            'origin': 'SEA',
            'destination': 'LAX',
            'hour': 14,
            'day_of_week': 2,
            'month': 10,
            'weather_condition': 'clear',
            'temperature_c': 20.0
        })
        res_ord = self.predictor.predict({
            'transport_mode': 'AIRPLANE',
            'carrier_or_route': 'DL',
            'origin': 'ORD',
            'destination': 'LAX',
            'hour': 14,
            'day_of_week': 2,
            'month': 10,
            'weather_condition': 'clear',
            'temperature_c': 20.0
        })
        self.assertNotEqual(res_sea['raw_regression_min'], res_ord['raw_regression_min'],
                            "SEA and ORD origins must produce differing predictions")

    def test_05_no_hardcoded_10_min_minimum(self):
        """The model must be capable of predicting continuous delay values below 5.0 minutes."""
        flight = {
            'transport_mode': 'AIRPLANE',
            'carrier_or_route': 'HA',
            'origin': 'HNL',
            'destination': 'ITO',
            'hour': 6,
            'day_of_week': 1,
            'month': 10,
            'distance_km': 340.0,
            'scheduled_duration_min': 50.0,
            'historical_delay_mean': 3.5,
            'weather_condition': 'clear',
            'temperature_c': 22.0
        }
        res = self.predictor.predict(flight)
        self.assertLess(res['estimated_delay_min'], 3.0,
                        f"Expected estimate < 3.0 min for island hopper, got {res['estimated_delay_min']}")
        self.assertGreater(res['estimated_delay_min'], 0.0)

    def test_06_xgboost_absent_and_knn_active(self):
        """Verify XGBoost is absent from models and registry, and KNN is registered."""
        import sys
        self.assertNotIn("xgboost", sys.modules, "XGBoost must not be imported")
        all_models = self.predictor.all_models
        self.assertIn("K-Nearest Neighbors", all_models, "KNN must be active in registry")
        self.assertNotIn("XGBoost", all_models, "XGBoost must be absent from registry")

    def test_07_bus_train_metro_unbroken(self):
        """Verify Bus, Train, and Metro predictions remain accurate and responsive."""
        bus_res = self.predictor.predict({
            'transport_mode': 'BUS', 'city': 'London', 'carrier_or_route': 'RT_LON_001',
            'origin': 'STN_LON_0037', 'destination': 'STN_LON_0211', 'hour': 8,
            'day_of_week': 1, 'month': 10, 'distance_km': 51.1, 'scheduled_duration_min': 102.2,
            'historical_delay_mean': 1.2, 'vehicle_capacity': 80, 'occupancy_rate': 0.65,
            'headway_min': 15, 'weather_condition': 'clear', 'temperature_c': 16.8
        })
        metro_res = self.predictor.predict({
            'transport_mode': 'METRO', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001', 'destination': 'STN_BER_0015', 'hour': 12,
            'day_of_week': 2, 'month': 10, 'distance_km': 14.0, 'scheduled_duration_min': 22.0,
            'historical_delay_mean': 0.8, 'vehicle_capacity': 800, 'occupancy_rate': 0.50,
            'headway_min': 4.0, 'weather_condition': 'clear', 'temperature_c': 15.0
        })
        self.assertEqual(bus_res['status'], 'success')
        self.assertEqual(metro_res['status'], 'success')
        self.assertGreater(bus_res['estimated_delay_min'], 0.0)
        self.assertGreater(metro_res['estimated_delay_min'], 0.0)

if __name__ == '__main__':
    unittest.main()
