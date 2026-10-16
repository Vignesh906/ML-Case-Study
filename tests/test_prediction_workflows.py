"""
Comprehensive Tests for Multimodal Prediction Page & Dynamic Route/Location Dropdowns
-------------------------------------------------------------------------------------
Verifies:
1. /api/routes-metadata endpoint returns genuine dataset values
2. Bus workflow: Mode -> City -> Route -> Origin -> Destination -> Prediction
3. Train/Rail workflow: Mode -> City -> Route -> Origin -> Destination -> Prediction
4. Metro workflow: Mode -> City -> Route -> Origin -> Destination -> Prediction
5. Airplane workflow: Mode -> Airline -> Origin Airport -> Destination Airport -> Prediction
6. Server-side validation rejecting invalid / identical origin-destination combinations
"""

import os
import sys
import unittest
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

class TestPredictionWorkflows(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_routes_metadata_endpoint(self):
        """Verifies /api/routes-metadata returns valid structure derived from real datasets."""
        res = self.client.get('/api/routes-metadata')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Check top-level keys
        self.assertIn('modes', data)
        self.assertIn('transit', data)
        self.assertIn('airplane', data)

        # Check supported modes
        self.assertListEqual(data['modes'], ['BUS', 'TRAIN', 'METRO', 'AIRPLANE'])

        # Check transit data
        for mode in ['BUS', 'TRAIN', 'METRO']:
            self.assertIn(mode, data['transit'])
            cities = data['transit'][mode]['cities']
            self.assertGreater(len(cities), 0)
            for city_name, city_info in cities.items():
                self.assertIn('routes', city_info)
                self.assertGreater(len(city_info['routes']), 0)
                # Check route properties
                route = city_info['routes'][0]
                self.assertTrue(route['route_id'].startswith('RT_'))
                self.assertTrue(route['origin'].startswith('STN_'))
                self.assertTrue(route['destination'].startswith('STN_'))
                self.assertGreater(route['dist_km'], 0)
                self.assertGreater(route['dur_min'], 0)
                self.assertGreater(route['capacity'], 0)

        # Check airplane data
        airlines = data['airplane']['airlines']
        self.assertEqual(len(airlines), 14)
        airports = data['airplane']['airports']
        self.assertEqual(len(airports), 322)
        flight_tree = data['airplane']['flight_tree']
        self.assertIn('DL', flight_tree)
        self.assertIn('ATL', flight_tree['DL'])
        self.assertIn('LAX', flight_tree['DL']['ATL'])

    def test_bus_workflow(self):
        """Test Bus workflow: Mode -> City -> Route -> Origin -> Destination -> Prediction"""
        meta_res = self.client.get('/api/routes-metadata')
        meta = meta_res.get_json()
        bus_routes = meta['transit']['BUS']['cities']['London']['routes']
        route = bus_routes[0]

        payload = {
            'transport_mode': 'BUS',
            'city': 'London',
            'carrier_or_route': route['route_id'],
            'origin': route['origin'],
            'destination': route['destination'],
            'hour': 8,
            'day_of_week': 1,
            'month': 10,
            'distance_km': route['dist_km'],
            'scheduled_duration_min': route['dur_min'],
            'historical_delay_mean': 2.5,
            'headway_min': route['headway_min'],
            'vehicle_capacity': route['capacity'],
            'occupancy_rate': 0.65,
            'weather_condition': 'clear',
            'temperature_c': 18.0
        }

        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data['status'], 'success')
        self.assertEqual(json_data['transport_mode'], 'BUS')
        self.assertIn(json_data['predicted_category'], ['ON TIME', 'MODERATELY DELAYED', 'SEVERELY DELAYED'])
        self.assertIn('probabilities', json_data)

    def test_train_workflow(self):
        """Test Train workflow: Mode -> City -> Route -> Origin -> Destination -> Prediction"""
        meta_res = self.client.get('/api/routes-metadata')
        meta = meta_res.get_json()
        train_routes = meta['transit']['TRAIN']['cities']['Berlin']['routes']
        route = train_routes[0]

        payload = {
            'transport_mode': 'TRAIN',
            'city': 'Berlin',
            'carrier_or_route': route['route_id'],
            'origin': route['origin'],
            'destination': route['destination'],
            'hour': 14,
            'day_of_week': 2,
            'month': 5,
            'distance_km': route['dist_km'],
            'scheduled_duration_min': route['dur_min'],
            'historical_delay_mean': 3.0,
            'headway_min': route['headway_min'],
            'vehicle_capacity': route['capacity'],
            'occupancy_rate': 0.70,
            'weather_condition': 'cloudy',
            'temperature_c': 20.0
        }

        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data['status'], 'success')
        self.assertEqual(json_data['transport_mode'], 'TRAIN')

    def test_metro_workflow(self):
        """Test Metro workflow: Mode -> City -> Route -> Origin -> Destination -> Prediction"""
        meta_res = self.client.get('/api/routes-metadata')
        meta = meta_res.get_json()
        metro_routes = meta['transit']['METRO']['cities']['Paris']['routes']
        route = metro_routes[0]

        payload = {
            'transport_mode': 'METRO',
            'city': 'Paris',
            'carrier_or_route': route['route_id'],
            'origin': route['origin'],
            'destination': route['destination'],
            'hour': 18,
            'day_of_week': 4,
            'month': 9,
            'distance_km': route['dist_km'],
            'scheduled_duration_min': route['dur_min'],
            'historical_delay_mean': 1.5,
            'headway_min': route['headway_min'],
            'vehicle_capacity': route['capacity'],
            'occupancy_rate': 0.85,
            'weather_condition': 'clear',
            'temperature_c': 22.0
        }

        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data['status'], 'success')
        self.assertEqual(json_data['transport_mode'], 'METRO')

    def test_airplane_workflow(self):
        """Test Airplane workflow: Mode -> Airline -> Origin Airport -> Destination Airport -> Prediction"""
        meta_res = self.client.get('/api/routes-metadata')
        meta = meta_res.get_json()
        flight_tree = meta['airplane']['flight_tree']

        # Delta from Atlanta to Los Angeles
        airline = 'DL'
        origin = 'ATL'
        dest = 'LAX'
        flight_info = flight_tree[airline][origin][dest]

        payload = {
            'transport_mode': 'AIRPLANE',
            'city': 'Atlanta',
            'carrier_or_route': airline,
            'origin': origin,
            'destination': dest,
            'hour': 11,
            'day_of_week': 3,
            'month': 7,
            'distance_km': flight_info['dist_km'],
            'scheduled_duration_min': flight_info['dur_min'],
            'historical_delay_mean': 10.9,
            'headway_min': 0,
            'vehicle_capacity': 180,
            'occupancy_rate': 0.82,
            'weather_condition': 'clear',
            'temperature_c': 25.0
        }

        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data['status'], 'success')
        self.assertEqual(json_data['transport_mode'], 'AIRPLANE')
        self.assertIn('probabilities', json_data)

    def test_validation_identical_origin_dest(self):
        """Rejects request where origin and destination are identical."""
        payload = {
            'transport_mode': 'AIRPLANE',
            'carrier_or_route': 'AA',
            'origin': 'JFK',
            'destination': 'JFK',
            'distance_km': 100,
            'scheduled_duration_min': 50
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("identical", res.get_json()['message'])

    def test_validation_invalid_mode(self):
        """Rejects unsupported transport mode."""
        payload = {
            'transport_mode': 'HELICOPTER',
            'carrier_or_route': 'HELI_1',
            'origin': 'PAD_1',
            'destination': 'PAD_2'
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid transport mode", res.get_json()['message'])

if __name__ == "__main__":
    unittest.main()
