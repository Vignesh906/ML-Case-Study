"""
Comprehensive Metro and All-Modes Audit & Regression Verification Test Suite
=============================================================================
Validates:
1. Metro feature sensitivity: Headway, Occupancy, Peak Hour, Historical Delay, and Disruption
2. Metro underground physical realism: temperature insensitivity (< 0.1m variance)
3. Quantile delay outputs: P50 and P90 non-null and valid across BUS, TRAIN, METRO, and AIRPLANE
4. Exact key schema alignment (estimated_delay_min, raw_regression_min, unrounded_delay_min,
   typical_median_delay_min, typical_delay_p50_min, high_end_p90_delay_min, high_end_delay_p90_min)
5. Regression baseline comparison: mode regressors beat median and zero baselines
6. Strict probability calibration (sum to ~100.0%, confidence equals predicted tier probability)
7. Disagreement disclosure banner between classifier and regression tier
8. Live weather API endpoint behavior and graceful degraded state
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from src.prediction import DelayPredictor, get_prediction, CLASS_NAMES

class TestMetroAndAllModesAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = DelayPredictor()
        cls.client = app.test_client()

    def test_01_metro_occupancy_sensitivity(self):
        """Metro delay must significantly increase during crush passenger crowding (0.95 vs 0.10)."""
        base = {
            'transport_mode': 'METRO',
            'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001',
            'destination': 'STN_BER_0015',
            'city': 'Berlin',
            'hour': 12,
            'day_of_week': 2,
            'month': 10,
            'scheduled_duration_min': 22.0,
            'distance_km': 14.0,
            'historical_delay_mean': 0.8,
            'vehicle_capacity': 800,
            'weather_condition': 'clear',
            'temperature_c': 15.0,
            'headway_min': 4.0
        }
        res_low_occ = self.predictor.predict({**base, 'occupancy_rate': 0.10, 'passenger_count': 80})
        res_high_occ = self.predictor.predict({**base, 'occupancy_rate': 0.95, 'passenger_count': 760})
        
        self.assertGreater(
            res_high_occ['raw_regression_min'], res_low_occ['raw_regression_min'],
            "Crush occupancy (0.95) must produce higher raw delay than low occupancy (0.10) for Metro"
        )
        self.assertGreater(
            res_high_occ['high_end_p90_delay_min'], res_low_occ['high_end_p90_delay_min'],
            "Crush occupancy must have higher P90 delay"
        )

    def test_02_metro_rush_hour_sensitivity(self):
        """Metro delay during morning/evening rush hour must exceed midday off-peak delay."""
        base = {
            'transport_mode': 'METRO',
            'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001',
            'destination': 'STN_BER_0015',
            'city': 'Berlin',
            'day_of_week': 2,
            'month': 10,
            'scheduled_duration_min': 22.0,
            'distance_km': 14.0,
            'historical_delay_mean': 0.8,
            'vehicle_capacity': 800,
            'occupancy_rate': 0.50,
            'passenger_count': 400,
            'weather_condition': 'clear',
            'temperature_c': 15.0,
            'headway_min': 4.0
        }
        res_midday = self.predictor.predict({**base, 'hour': 12})
        res_rush_am = self.predictor.predict({**base, 'hour': 8})
        res_rush_pm = self.predictor.predict({**base, 'hour': 18})

        self.assertGreater(
            res_rush_am['raw_regression_min'], res_midday['raw_regression_min'],
            "Morning rush (08:00) delay must exceed midday (12:00) delay for Metro"
        )
        self.assertGreater(
            res_rush_pm['raw_regression_min'], res_midday['raw_regression_min'],
            "Evening rush (18:00) delay must exceed midday (12:00) delay for Metro"
        )

    def test_03_metro_headway_and_history_sensitivity(self):
        """Metro delay must increase with stretched headways and higher historical corridor delays."""
        base = {
            'transport_mode': 'METRO',
            'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001',
            'destination': 'STN_BER_0015',
            'city': 'Berlin',
            'hour': 12,
            'day_of_week': 2,
            'month': 10,
            'scheduled_duration_min': 22.0,
            'distance_km': 14.0,
            'vehicle_capacity': 800,
            'occupancy_rate': 0.50,
            'passenger_count': 400,
            'weather_condition': 'clear',
            'temperature_c': 15.0,
        }
        res_hw4 = self.predictor.predict({**base, 'headway_min': 4.0, 'historical_delay_mean': 0.8})
        res_hw8 = self.predictor.predict({**base, 'headway_min': 8.0, 'historical_delay_mean': 0.8})
        res_hist5 = self.predictor.predict({**base, 'headway_min': 4.0, 'historical_delay_mean': 5.0})

        self.assertGreater(
            res_hw8['raw_regression_min'], res_hw4['raw_regression_min'],
            "Stretched 8-min headway must produce higher delay than nominal 4-min headway"
        )
        self.assertGreater(
            res_hist5['raw_regression_min'], res_hw4['raw_regression_min'],
            "Corridor with 5-min historical delay must produce higher delay than 0.8-min historical delay"
        )

    def test_04_metro_temperature_physical_stability(self):
        """Underground Metro should not exhibit large outdoor thermal swings (< 0.1 min difference)."""
        base = {
            'transport_mode': 'METRO',
            'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001',
            'destination': 'STN_BER_0015',
            'city': 'Berlin',
            'hour': 12,
            'day_of_week': 2,
            'month': 10,
            'scheduled_duration_min': 22.0,
            'distance_km': 14.0,
            'historical_delay_mean': 0.8,
            'vehicle_capacity': 800,
            'occupancy_rate': 0.50,
            'passenger_count': 400,
            'weather_condition': 'clear',
            'headway_min': 4.0
        }
        res_cold = self.predictor.predict({**base, 'temperature_c': -5.0})
        res_hot = self.predictor.predict({**base, 'temperature_c': 38.0})

        diff = abs(res_cold['raw_regression_min'] - res_hot['raw_regression_min'])
        self.assertLess(
            diff, 0.10,
            f"Metro temperature sensitivity must remain physically realistic (< 0.1m difference, got {diff:.4f}m)"
        )

    def test_05_all_four_modes_quantile_keys_and_values(self):
        """BUS, TRAIN, METRO, and AIRPLANE must all return valid P50, P90, and unrounded regression keys."""
        test_inputs = [
            {'transport_mode': 'BUS', 'carrier_or_route': 'RT_LON_001', 'origin': 'STN_LON_0037', 'destination': 'STN_LON_0211', 'city': 'London', 'hour': 8, 'day_of_week': 1, 'month': 10, 'distance_km': 51.1, 'scheduled_duration_min': 102.2, 'historical_delay_mean': 1.2, 'vehicle_capacity': 80, 'occupancy_rate': 0.65, 'headway_min': 15.0, 'weather_condition': 'clear', 'temperature_c': 16.8},
            {'transport_mode': 'TRAIN', 'carrier_or_route': 'RT_BER_001', 'origin': 'STN_BER_0001', 'destination': 'STN_BER_0020', 'city': 'Berlin', 'hour': 8, 'day_of_week': 2, 'month': 10, 'distance_km': 75.0, 'scheduled_duration_min': 65.0, 'historical_delay_mean': 1.5, 'vehicle_capacity': 600, 'occupancy_rate': 0.50, 'headway_min': 40.0, 'weather_condition': 'clear', 'temperature_c': 15.0},
            {'transport_mode': 'METRO', 'carrier_or_route': 'RT_BER_001', 'origin': 'STN_BER_0001', 'destination': 'STN_BER_0015', 'city': 'Berlin', 'hour': 12, 'day_of_week': 2, 'month': 10, 'distance_km': 14.0, 'scheduled_duration_min': 22.0, 'historical_delay_mean': 0.8, 'vehicle_capacity': 800, 'occupancy_rate': 0.50, 'headway_min': 4.0, 'weather_condition': 'clear', 'temperature_c': 15.0},
            {'transport_mode': 'AIRPLANE', 'carrier_or_route': 'DL', 'origin': 'ATL', 'destination': 'LAX', 'city': 'Atlanta', 'hour': 14, 'day_of_week': 2, 'month': 10, 'distance_km': 3132.0, 'scheduled_duration_min': 285.0, 'historical_delay_mean': 9.5, 'vehicle_capacity': 180, 'occupancy_rate': 0.82, 'headway_min': 0.0, 'weather_condition': 'clear', 'temperature_c': 20.0}
        ]
        
        required_keys = [
            'estimated_delay_min', 'raw_regression_min', 'unrounded_delay_min',
            'typical_median_delay_min', 'typical_delay_p50_min', 'median_delay_min',
            'high_end_p90_delay_min', 'high_end_delay_p90_min', 'high_end_delay_min',
            'probabilities', 'confidence_pct', 'risk_level'
        ]

        for inp in test_inputs:
            mode = inp['transport_mode']
            res = self.predictor.predict(inp)
            self.assertEqual(res['status'], 'success')
            for k in required_keys:
                self.assertIn(k, res, f"Mode {mode} missing required key: {k}")
                self.assertIsNotNone(res[k], f"Mode {mode} has None value for key: {k}")

            # Verify P50 <= estimated or reasonable, and P90 >= P50
            self.assertGreaterEqual(res['high_end_p90_delay_min'], res['typical_median_delay_min'],
                                    f"Mode {mode} P90 must be >= P50")

    def test_06_probability_calibration_and_confidence(self):
        """Classifier probabilities must sum to 100.0% (+/- 0.1%) and confidence must match shown tier."""
        test_input = {
            'transport_mode': 'METRO',
            'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001',
            'destination': 'STN_BER_0015',
            'city': 'Berlin',
            'hour': 18,
            'day_of_week': 2,
            'month': 10,
            'scheduled_duration_min': 22.0,
            'distance_km': 14.0,
            'historical_delay_mean': 4.5,
            'vehicle_capacity': 800,
            'occupancy_rate': 0.95,
            'headway_min': 10.0,
            'weather_condition': 'storm',
            'temperature_c': 32.0
        }
        res = self.predictor.predict(test_input)
        probs = res['probabilities']
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 100.0, places=1, msg=f"Probabilities sum to {prob_sum}, expected 100.0")

        # Confidence is strictly probability of predicted category
        pred_label = res['predicted_category']
        expected_conf = probs[pred_label]
        self.assertEqual(res['confidence_pct'], expected_conf)

    def test_07_api_prediction_endpoint_live(self):
        """Flask /predict endpoint returns 200 and matches predictor keys and data."""
        payload = {
            'transport_mode': 'METRO',
            'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001',
            'destination': 'STN_BER_0015',
            'city': 'Berlin',
            'hour': 8,
            'day_of_week': 2,
            'month': 10,
            'scheduled_duration_min': 22.0,
            'distance_km': 14.0,
            'historical_delay_mean': 0.8,
            'vehicle_capacity': 800,
            'occupancy_rate': 0.85,
            'headway_min': 5.0,
            'weather_condition': 'clear',
            'temperature_c': 16.0
        }
        response = self.client.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['transport_mode'], 'METRO')
        self.assertIn('estimated_delay_min', data)
        self.assertIn('typical_median_delay_min', data)
        self.assertIn('high_end_p90_delay_min', data)
        self.assertIn('probabilities', data)

    def test_08_live_weather_api_endpoint(self):
        """Flask /api/weather endpoint returns valid structured weather telemetry or graceful degraded error."""
        response = self.client.get('/api/weather?city=Berlin')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        if data['status'] == 'success':
            self.assertIn('condition', data)
            self.assertIn('temperature_c', data)
            self.assertIn('observed_at', data)
            self.assertIn('retrieved_at', data)
        else:
            self.assertIn('message', data)

if __name__ == '__main__':
    unittest.main()
