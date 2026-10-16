"""
Comprehensive Audit & Regression Test Suite
============================================
Validates all 14 required audit scenarios:
1.  Dataset loading and schema validation
2.  Model loading and prediction
3.  Feature order and preprocessing consistency
4.  Repeated identical requests (Test A - Reproducibility)
5.  One-feature-at-a-time changes (Test B - Sensitivity)
6.  All four transportation modes (Test C - Multi-feature variation)
7.  Severity and risk consistency (no contradictions)
8.  Probability-sum validation (exact 100.0%)
9.  Weather API failure and graceful fallback
10. Dashboard metrics and evaluation split verification
11. Prediction history accuracy
12. Missing, invalid, out-of-range, and non-numeric inputs
13. XGBoost removal and KNN registration confirmation
14. Model serialization and application restart
"""

import os
import sys
import unittest
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from src.prediction import DelayPredictor, predictor_instance, CLASS_NAMES
from src.preprocessing import MultimodalPreprocessor, load_preprocessor
from src.analytics_data import get_dataset_stats

class TestAuditRegressionSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = DelayPredictor()
        cls.client = app.test_client()

    # 1. Dataset loading and schema validation
    def test_01_dataset_loading_and_schema(self):
        """Verify train, val, and test datasets exist, load cleanly, and have valid schemas."""
        for split in ['train', 'val', 'test']:
            path = os.path.join(BASE_DIR, "data", "processed", f"multimodal_{split}.csv")
            self.assertTrue(os.path.exists(path), f"Missing dataset file: {path}")
            df = pd.read_csv(path, nrows=50)
            required_cols = [
                'transport_mode', 'hour', 'day_of_week', 'month', 'day',
                'is_weekend', 'is_peak_hour', 'distance_km', 'scheduled_duration_min',
                'carrier_or_route', 'origin', 'destination', 'city',
                'historical_delay_mean', 'vehicle_capacity', 'occupancy_rate',
                'passenger_count', 'weather_condition', 'temperature_c',
                'headway_min', 'delay_minutes', 'delay_category'
            ]
            for col in required_cols:
                self.assertIn(col, df.columns, f"Split {split} missing required column: {col}")
            # Target validation
            self.assertTrue(df['delay_category'].isin([0, 1, 2]).all())
            self.assertTrue((df['delay_minutes'] >= 0.0).all())

    # 2. Model loading and prediction
    def test_02_model_loading_and_prediction(self):
        """Verify best model, delay regressors, and preprocessing pipeline load and predict successfully."""
        self.assertTrue(self.predictor.is_ready())
        self.assertIn(self.predictor.model_name, ["Linear SVM (calibrated)", "Random Forest", "K-Nearest Neighbors"])
        self.assertIsNotNone(self.predictor.best_model)
        self.assertIsNotNone(self.predictor.mode_regressors)
        self.assertIn('BUS', self.predictor.mode_regressors)
        self.assertIn('AIRPLANE', self.predictor.mode_regressors)

    # 3. Feature order and preprocessing consistency
    def test_03_feature_order_and_preprocessing_consistency(self):
        """Verify pipeline output dimensionality and feature order match trained model expectations."""
        prep_path = os.path.join(BASE_DIR, "models", "preprocessing_pipeline.joblib")
        preprocessor = load_preprocessor(prep_path)
        feature_names = preprocessor.get_feature_names()
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 30)

        # Check model comparison feature names match pipeline
        metrics_path = os.path.join(BASE_DIR, "results", "metrics", "model_comparison.json")
        with open(metrics_path, "r") as f:
            comp = json.load(f)
        self.assertEqual(feature_names, comp["feature_names"])

    # 4. Repeated identical requests (Test A - Reproducibility)
    def test_04_test_a_reproducibility(self):
        """Verify submitting identical inputs produces deterministic and identical predictions."""
        payload = {
            'transport_mode': 'BUS',
            'city': 'London',
            'carrier_or_route': 'ROUTE_LON_01',
            'origin': 'STN_LON_0012',
            'destination': 'STN_LON_0037',
            'historical_delay_mean': 1.1,
            'hour': 10,
            'day_of_week': 3,
            'occupancy_rate': 0.70,
            'headway_min': 10.0,
            'temperature_c': 18.0,
            'weather_condition': 'clear'
        }
        res1 = self.predictor.predict(payload)
        res2 = self.predictor.predict(payload)
        res3 = self.predictor.predict(payload)

        self.assertEqual(res1['predicted_category'], res2['predicted_category'])
        self.assertEqual(res2['predicted_category'], res3['predicted_category'])
        self.assertAlmostEqual(res1['estimated_delay_min'], res2['estimated_delay_min'], places=4)
        self.assertAlmostEqual(res2['estimated_delay_min'], res3['estimated_delay_min'], places=4)
        self.assertEqual(res1['probabilities'], res2['probabilities'])
        self.assertEqual(res1['confidence_pct'], res2['confidence_pct'])

    # 5. One-feature-at-a-time changes (Test B - Sensitivity)
    def test_05_test_b_one_feature_sensitivity(self):
        """Verify changing one significant operational parameter propagates to model features and outputs."""
        base_flight = {
            'transport_mode': 'AIRPLANE',
            'city': 'Atlanta',
            'carrier_or_route': 'DL',
            'origin': 'ATL',
            'destination': 'ORD',
            'historical_delay_mean': 10.9,
            'hour': 12,
            'day_of_week': 2,
            'distance_km': 975.0,
            'scheduled_duration_min': 120.0
        }
        high_hist_flight = dict(base_flight, historical_delay_mean=35.0)

        res_base = self.predictor.predict(base_flight)
        res_high = self.predictor.predict(high_hist_flight)

        # Baseline delay sensitivity
        self.assertGreater(res_high['estimated_delay_min'], res_base['estimated_delay_min'])
        self.assertGreater(
            res_high['probabilities']['SEVERELY DELAYED'],
            res_base['probabilities']['SEVERELY DELAYED']
        )

    # 6. All four transportation modes (Test C - Multi-feature variation)
    def test_06_test_c_multi_feature_all_modes(self):
        """Verify all 4 transportation modes handle complex multi-feature payloads correctly."""
        modes = ['BUS', 'TRAIN', 'METRO', 'AIRPLANE']
        for mode in modes:
            payload = {
                'transport_mode': mode,
                'city': 'Chicago' if mode != 'BUS' else 'London',
                'carrier_or_route': 'UA' if mode == 'AIRPLANE' else 'ROUTE_01',
                'origin': 'ORD' if mode == 'AIRPLANE' else 'STN_A',
                'destination': 'DEN' if mode == 'AIRPLANE' else 'STN_B',
                'hour': 17,
                'day_of_week': 4,
                'occupancy_rate': 0.85,
                'historical_delay_mean': 2.0 if mode != 'AIRPLANE' else 18.0,
                'headway_min': 8.0 if mode != 'AIRPLANE' else 0.0,
                'weather_condition': 'light_rain'
            }
            res = self.predictor.predict(payload)
            self.assertEqual(res['status'], 'success')
            self.assertEqual(res['transport_mode'], mode)
            self.assertIn(res['predicted_category'], CLASS_NAMES)
            self.assertIsInstance(res['estimated_delay_min'], float)

    # 7. Severity and risk consistency
    def test_07_severity_and_risk_consistency(self):
        """Verify severity tiers, continuous delays, and risk levels do not contradict documented thresholds."""
        test_delays = [
            ({'historical_delay_mean': 0.5, 'transport_mode': 'METRO'}, 'ON TIME', 'Low'),
            ({'historical_delay_mean': 12.0, 'transport_mode': 'AIRPLANE', 'hour': 14}, 'MODERATELY DELAYED', 'Moderate')
        ]
        for inputs, expected_cat, expected_risk in test_delays:
            res = self.predictor.predict(inputs)
            self.assertEqual(res['predicted_category'], expected_cat)
            self.assertEqual(res['risk_level'], expected_risk)
            if expected_cat == 'ON TIME':
                self.assertLessEqual(res['estimated_delay_min'], 5.0)
                self.assertIn('<= 5 min', res['risk_desc'])
            elif expected_cat == 'MODERATELY DELAYED':
                self.assertGreater(res['estimated_delay_min'], 5.0)
                self.assertLessEqual(res['estimated_delay_min'], 20.0)
                self.assertIn('> 5 to 20 min', res['risk_desc'])

    # 8. Probability-sum validation
    def test_08_probability_sum_validation(self):
        """Verify class probabilities always sum exactly to 100.0% and confidence matches predicted class."""
        for mode in ['BUS', 'TRAIN', 'METRO', 'AIRPLANE']:
            res = self.predictor.predict({'transport_mode': mode, 'hour': 15})
            probs = res['probabilities']
            total = sum(probs.values())
            self.assertAlmostEqual(total, 100.0, places=1)
            pred_class = res['predicted_category']
            self.assertEqual(res['confidence_pct'], probs[pred_class])

    # 9. Weather API failure and fallback
    def test_09_weather_api_failure_and_fallback(self):
        """Verify /api/live-weather endpoint responds safely and falls back cleanly without crashing."""
        res = self.client.get('/api/live-weather?target=UNKNOWN_LOCATION')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data['status'], ['success', 'fallback'])
        self.assertIn('temperature_c', data)
        self.assertIn('weather_condition', data)
        self.assertIn('source', data)

    # 10. Dashboard metrics and evaluation split
    def test_10_dashboard_metrics_and_evaluation_split(self):
        """Verify dashboard statistics match physical dataset row counts (104,000 total)."""
        stats = get_dataset_stats()
        self.assertEqual(stats['total_records'], 104000)
        self.assertEqual(stats['train_records'], 80000)
        self.assertEqual(stats['val_records'], 12000)
        self.assertEqual(stats['test_records'], 12000)
        self.assertEqual(len(stats['modes']), 4)
        for mode, count in stats['modes'].items():
            self.assertEqual(count, 20000)

    # 11. Prediction history accuracy
    def test_11_prediction_history_accuracy(self):
        """Verify response contains all fields needed for complete and accurate prediction history logging."""
        payload = {
            'transport_mode': 'TRAIN',
            'city': 'Paris',
            'carrier_or_route': 'RT_PAR_001',
            'origin': 'STN_PAR_0001',
            'destination': 'STN_PAR_0010',
            'hour': 12
        }
        res = self.predictor.predict(payload)
        required_fields = [
            'prediction_timestamp', 'transport_mode', 'predicted_category',
            'estimated_delay_min', 'confidence_pct', 'risk_level', 'risk_badge'
        ]
        for field in required_fields:
            self.assertIn(field, res)
            self.assertIsNotNone(res[field])

    # 12. Missing, invalid, out-of-range, and non-numeric inputs
    def test_12_missing_and_extreme_inputs(self):
        """Verify predictor defensively handles empty, non-numeric, and extreme inputs without unhandled exceptions."""
        empty_res = self.predictor.predict({})
        self.assertEqual(empty_res['status'], 'success')
        self.assertIn(empty_res['predicted_category'], CLASS_NAMES)

        extreme_res = self.predictor.predict({
            'transport_mode': 'INVALID_MODE',
            'hour': 'not_an_hour',
            'day_of_week': -50,
            'distance_km': 'huge',
            'occupancy_rate': 999.0,
            'temperature_c': -999.0
        })
        self.assertEqual(extreme_res['status'], 'success')
        self.assertIn(extreme_res['predicted_category'], CLASS_NAMES)

    # 13. XGBoost removal and KNN registration confirmation
    def test_13_xgboost_removal_and_knn_registration(self):
        """Confirm XGBoost is absent from all active models and KNN is registered with legitimate metrics."""
        metrics_path = os.path.join(BASE_DIR, "results", "metrics", "model_comparison.json")
        with open(metrics_path, "r") as f:
            comp = json.load(f)

        models = comp.get("models", {})
        self.assertNotIn("XGBoost", models)
        self.assertIn("K-Nearest Neighbors", models)
        self.assertIn(comp.get("best_model"), ["Linear SVM (calibrated)", "Random Forest"])
        self.assertGreater(comp["models"]["K-Nearest Neighbors"]["f1_macro"], 0.30)

    # 14. Model serialization and application restart
    def test_14_model_serialization_and_restart(self):
        """Verify models can be freshly deserialized from disk simulating a server restart."""
        new_predictor = DelayPredictor()
        self.assertTrue(new_predictor.is_ready())
        self.assertIn(new_predictor.model_name, ["Linear SVM (calibrated)", "Random Forest", "K-Nearest Neighbors"])
        res = new_predictor.predict({'transport_mode': 'METRO', 'hour': 9})
        self.assertEqual(res['status'], 'success')

    # 15. Screenshot scenario verification: Bus delay with potential classifier disagreement
    def test_15_reproduce_bus_screenshot_disagreement_case(self):
        """Reproduce the Bus screenshot case:
        Estimated delay ~1.02-1.52 min, classifier boundary probabilities.
        Verify documented policy:
        1. Estimated delay <= 5.0 min derives ON TIME severity tier.
        2. Operational risk is Low Risk (0-5 min delay expected).
        3. Confidence is strictly the probability of the predicted severity tier.
        4. When classifier argmax differs from regression tier, disagreement is explicitly disclosed.
        5. Probabilities sum to exactly 100.0%.
        """
        payload = {
            'transport_mode': 'BUS',
            'city': 'London',
            'carrier_or_route': 'ROUTE_LON_01',
            'origin': 'STN_LON_0012',
            'destination': 'STN_LON_0037',
            'historical_delay_mean': 1.1,
            'hour': 10,
            'day_of_week': 3,
            'occupancy_rate': 0.65,
            'headway_min': 16.0,
            'vehicle_capacity': 80.0,
            'temperature_c': 12.4,
            'weather_condition': 'clear'
        }
        # Bus nominal case
        res = self.predictor.predict(payload)
        self.assertEqual(res['status'], 'success')
        self.assertLessEqual(res['estimated_delay_min'], 5.0)
        self.assertEqual(res['predicted_category'], 'ON TIME')
        self.assertEqual(res['risk_level'], 'Low')
        self.assertEqual(res['confidence_pct'], res['probabilities']['ON TIME'])
        self.assertAlmostEqual(sum(res['probabilities'].values()), 100.0, places=1)
        if res['has_disagreement']:
            self.assertIsNotNone(res['disagreement_disclosure'])
            self.assertIn('Model Disagreement Disclosed', res['disagreement_disclosure'])

        # Explicit model disagreement verification:
        # High-variance flight boundary where continuous regression (7.61 min -> MODERATELY DELAYED)
        # and balanced classifier (SEVERELY DELAYED) diverge.
        payload_disagree = {
            'transport_mode': 'AIRPLANE',
            'city': 'Orlando',
            'carrier_or_route': 'B6',
            'origin': 'MCO',
            'destination': 'PSE',
            'historical_delay_mean': 18.5,
            'hour': 21,
            'day_of_week': 4,
            'occupancy_rate': 0.82,
            'headway_min': 0.0,
            'vehicle_capacity': 180.0,
            'temperature_c': 8.0,
            'weather_condition': 'clear'
        }
        res_disagree = self.predictor.predict(payload_disagree)
        self.assertEqual(res_disagree['status'], 'success')
        self.assertTrue(res_disagree['has_disagreement'])
        self.assertIsNotNone(res_disagree['disagreement_disclosure'])
        self.assertIn('Model Disagreement Disclosed', res_disagree['disagreement_disclosure'])
        self.assertIn('Per documented prediction policy', res_disagree['disagreement_disclosure'])

if __name__ == '__main__':
    unittest.main()
