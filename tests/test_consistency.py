import unittest
import numpy as np
from src.prediction import DelayPredictor, CLASS_NAMES

class TestPredictionConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = DelayPredictor()

    def test_train_nominal_is_ontime_and_low_risk(self):
        """Verify Train with nominal inputs yields On Time, <= 5 min delay, and Low Risk."""
        payload = {
            'transport_mode': 'TRAIN',
            'city': 'Paris',
            'carrier_or_route': 'RT_PAR_001',
            'origin': 'STN_PAR_0001',
            'destination': 'STN_PAR_0010',
            'historical_delay_mean': 0.6,
            'hour': 12,
            'day_of_week': 2,
            'occupancy_rate': 0.65
        }
        res = self.predictor.predict(payload)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['predicted_category'], 'ON TIME')
        self.assertLessEqual(res['estimated_delay_min'], 5.0)
        self.assertEqual(res['risk_level'], 'Low')
        self.assertIn('<= 5 min', res['risk_desc'])
        
        # Probabilities validation
        probs = res['probabilities']
        self.assertAlmostEqual(sum(probs.values()), 100.0, places=1)
        self.assertEqual(res['confidence_pct'], probs['ON TIME'])
        self.assertGreaterEqual(probs['ON TIME'], probs['MODERATELY DELAYED'])
        self.assertGreaterEqual(probs['ON TIME'], probs['SEVERELY DELAYED'])

    def test_bus_nominal_is_ontime_and_low_risk(self):
        """Verify Bus with nominal inputs yields On Time, <= 5 min delay, and Low Risk (no moderate risk contradiction)."""
        payload = {
            'transport_mode': 'BUS',
            'city': 'London',
            'carrier_or_route': 'ROUTE_LON_01',
            'origin': 'STN_LON_0012',
            'destination': 'STN_LON_0037',
            'historical_delay_mean': 1.1,
            'hour': 10,
            'day_of_week': 3,
            'occupancy_rate': 0.70
        }
        res = self.predictor.predict(payload)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['predicted_category'], 'ON TIME')
        self.assertLessEqual(res['estimated_delay_min'], 5.0)
        self.assertEqual(res['risk_level'], 'Low')
        self.assertIn('<= 5 min', res['risk_desc'])
        
        # Probabilities validation
        probs = res['probabilities']
        self.assertAlmostEqual(sum(probs.values()), 100.0, places=1)
        self.assertEqual(res['confidence_pct'], probs['ON TIME'])

    def test_flight_moderate_delay_consistency(self):
        """Verify commercial flight with moderate variance maps consistently across regression, severity, and risk."""
        payload = {
            'transport_mode': 'AIRPLANE',
            'city': 'Atlanta',
            'carrier_or_route': 'DL',
            'origin': 'ATL',
            'destination': 'ORD',
            'historical_delay_mean': 10.9,
            'hour': 14,
            'day_of_week': 4
        }
        res = self.predictor.predict(payload)
        self.assertEqual(res['status'], 'success')
        self.assertGreater(res['estimated_delay_min'], 5.0)
        self.assertLessEqual(res['estimated_delay_min'], 20.0)
        self.assertEqual(res['predicted_category'], 'MODERATELY DELAYED')
        self.assertEqual(res['risk_level'], 'Moderate')
        self.assertIn('> 5 to 20 min', res['risk_desc'])
        self.assertEqual(res['confidence_pct'], res['probabilities']['MODERATELY DELAYED'])

    def test_flight_severe_delay_consistency(self):
        """Verify commercial flight with severe baseline maps to severe tier and high risk."""
        payload = {
            'transport_mode': 'AIRPLANE',
            'city': 'Chicago',
            'carrier_or_route': 'F9',
            'origin': 'ORD',
            'destination': 'DEN',
            'historical_delay_mean': 55.0,
            'hour': 21,
            'day_of_week': 5,
            'distance_km': 3500,
            'scheduled_duration_min': 320,
            'weather_condition': 'storm'
        }
        res = self.predictor.predict(payload)
        self.assertEqual(res['status'], 'success')
        self.assertGreater(res['estimated_delay_min'], 20.0)
        self.assertEqual(res['predicted_category'], 'SEVERELY DELAYED')
        self.assertEqual(res['risk_level'], 'High')
        self.assertIn('> 20 min', res['risk_desc'])
        self.assertEqual(res['confidence_pct'], res['probabilities']['SEVERELY DELAYED'])

    def test_metro_punctuality(self):
        """Verify rapid transit metro operates under high on-time punctuality."""
        payload = {
            'transport_mode': 'METRO',
            'city': 'Berlin',
            'carrier_or_route': 'RT_BER_001',
            'origin': 'STN_BER_0001',
            'destination': 'STN_BER_0005',
            'historical_delay_mean': 0.4,
            'hour': 9,
            'day_of_week': 1
        }
        res = self.predictor.predict(payload)
        self.assertEqual(res['predicted_category'], 'ON TIME')
        self.assertLessEqual(res['estimated_delay_min'], 5.0)
        self.assertEqual(res['risk_level'], 'Low')

if __name__ == '__main__':
    unittest.main()
