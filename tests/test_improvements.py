"""
Unit and Integration Tests for Project Improvements:
- Multi-stop route dropdowns and bidirectional corridor verification
- Live Open-Meteo meteorological telemetry endpoint (/api/live-weather)
- Continuous delay minutes regression estimation (/api/predict)
- Mode-specific held-out test evaluation metrics
"""

import os
import sys
import unittest
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

class TestProjectImprovements(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_multi_stop_metadata_structure(self):
        """Verifies that routes provide all stops and valid destinations mapping."""
        res = self.client.get('/api/routes-metadata')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        for mode in ['BUS', 'TRAIN', 'METRO']:
            for city_name, city_info in data['transit'][mode]['cities'].items():
                for route in city_info['routes']:
                    self.assertIn('stops', route)
                    self.assertIn('valid_destinations', route)
                    self.assertGreaterEqual(len(route['stops']), 1)
                    # For corridors with 2 terminal stops, both must be distinct
                    if len(route['stops']) > 1:
                        self.assertNotEqual(route['stops'][0], route['stops'][1])
                        # Check bidirectional mapping
                        st0 = route['stops'][0]
                        st1 = route['stops'][1]
                        self.assertIn(st1, route['valid_destinations'][st0])
                        self.assertIn(st0, route['valid_destinations'][st1])

    def test_live_weather_telemetry_endpoint(self):
        """Verifies that /api/live-weather responds safely with structured data."""
        res = self.client.get('/api/live-weather?target=New_York')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data['status'], ['success', 'fallback'])
        self.assertIn('temperature_c', data)
        self.assertIn('weather_condition', data)
        self.assertIn('source', data)
        self.assertIn('is_live', data)

    def test_live_weather_aviation_hub(self):
        """Verifies weather lookup for major airport hub (ATL)."""
        res = self.client.get('/api/live-weather?target=ATL')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data['status'], ['success', 'fallback'])
        self.assertIn('temperature_c', data)
        self.assertIn('Atlanta', data['location'])

    def test_continuous_estimated_delay_output(self):
        """Verifies /api/predict returns estimated_delay_min from regression model."""
        payload = {
            'transport_mode': 'BUS',
            'city': 'New_York',
            'carrier_or_route': 'ROUTE_01',
            'origin': 'STN_NEW_0136',
            'destination': 'STN_NEW_0045',
            'hour': 8,
            'day_of_week': 1,
            'month': 10,
            'distance_km': 15.5,
            'scheduled_duration_min': 45.0,
            'historical_delay_mean': 2.5,
            'headway_min': 10.0,
            'vehicle_capacity': 80,
            'occupancy_rate': 0.70,
            'weather_condition': 'clear',
            'temperature_c': 20.0
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('estimated_delay_min', data)
        self.assertIsInstance(data['estimated_delay_min'], (int, float))
        self.assertGreaterEqual(data['estimated_delay_min'], 0.0)

    def test_mode_stratified_metrics_exist(self):
        """Verifies model_comparison.json includes mode-specific evaluations and regression metrics."""
        metrics_path = os.path.join(BASE_DIR, "results", "metrics", "model_comparison.json")
        self.assertTrue(os.path.exists(metrics_path))
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        self.assertIn('mode_evaluations', data)
        self.assertIn('regression_metrics', data)
        best_model = data['best_model']
        self.assertIn(best_model, data['mode_evaluations'])
        for mode in ['BUS', 'TRAIN', 'METRO', 'AIRPLANE']:
            self.assertIn(mode, data['mode_evaluations'][best_model])
            mode_eval = data['mode_evaluations'][best_model][mode]
            self.assertEqual(mode_eval['sample_count'], 3000)
            self.assertIn('accuracy', mode_eval)
            self.assertIn('f1_macro', mode_eval)

if __name__ == '__main__':
    unittest.main()
