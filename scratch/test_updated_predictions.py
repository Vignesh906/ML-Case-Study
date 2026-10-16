import sys, os
BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prediction import DelayPredictor
p = DelayPredictor()

test_cases = [
    {'name': 'Bus Clear 16.8C', 'data': {'transport_mode': 'BUS', 'city': 'London', 'carrier_or_route': 'RT_LON_001', 'origin': 'STN_LON_0037', 'destination': 'STN_LON_0211', 'hour': 8, 'day_of_week': 1, 'month': 10, 'distance_km': 51.1, 'scheduled_duration_min': 102.2, 'historical_delay_mean': 1.2, 'headway_min': 15, 'occupancy_rate': 0.65, 'temperature_c': 16.8, 'weather_condition': 'clear'}},
    {'name': 'Bus Storm 16.8C', 'data': {'transport_mode': 'BUS', 'city': 'London', 'carrier_or_route': 'RT_LON_001', 'origin': 'STN_LON_0037', 'destination': 'STN_LON_0211', 'hour': 8, 'day_of_week': 1, 'month': 10, 'distance_km': 51.1, 'scheduled_duration_min': 102.2, 'historical_delay_mean': 1.2, 'headway_min': 15, 'occupancy_rate': 0.65, 'temperature_c': 16.8, 'weather_condition': 'storm'}},
    {'name': 'Train Clear 15.9C', 'data': {'transport_mode': 'TRAIN', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_002', 'origin': 'STN_BER_0094', 'destination': 'STN_BER_0127', 'hour': 8, 'day_of_week': 1, 'month': 10, 'distance_km': 12.7, 'scheduled_duration_min': 25.4, 'historical_delay_mean': 0.6, 'headway_min': 48.0, 'occupancy_rate': 0.65, 'temperature_c': 15.9, 'weather_condition': 'clear'}},
    {'name': 'Train Storm 40.9C', 'data': {'transport_mode': 'TRAIN', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_002', 'origin': 'STN_BER_0094', 'destination': 'STN_BER_0127', 'hour': 8, 'day_of_week': 1, 'month': 10, 'distance_km': 12.7, 'scheduled_duration_min': 25.4, 'historical_delay_mean': 0.6, 'headway_min': 48.0, 'occupancy_rate': 0.65, 'temperature_c': 40.9, 'weather_condition': 'storm'}},
    {'name': 'Metro Hour 8 (15.9C)', 'data': {'transport_mode': 'METRO', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_001', 'origin': 'STN_BER_0103', 'destination': 'STN_BER_0114', 'hour': 8, 'day_of_week': 1, 'month': 10, 'distance_km': 23.8, 'scheduled_duration_min': 47.6, 'historical_delay_mean': 0.4, 'headway_min': 9.0, 'occupancy_rate': 0.65, 'temperature_c': 15.9, 'weather_condition': 'clear'}},
    {'name': 'Metro Hour 12 (15.9C)', 'data': {'transport_mode': 'METRO', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_001', 'origin': 'STN_BER_0103', 'destination': 'STN_BER_0114', 'hour': 12, 'day_of_week': 1, 'month': 10, 'distance_km': 23.8, 'scheduled_duration_min': 47.6, 'historical_delay_mean': 0.4, 'headway_min': 9.0, 'occupancy_rate': 0.65, 'temperature_c': 15.9, 'weather_condition': 'clear'}},
    {'name': 'Flight Clear (DL ATL->LAX)', 'data': {'transport_mode': 'AIRPLANE', 'city': 'Atlanta', 'carrier_or_route': 'DL', 'origin': 'ATL', 'destination': 'LAX', 'hour': 12, 'day_of_week': 1, 'month': 10, 'distance_km': 3132.0, 'scheduled_duration_min': 300.0, 'historical_delay_mean': 11.2, 'temperature_c': 15.0, 'weather_condition': 'clear'}},
    {'name': 'Flight Storm (DL ATL->LAX)', 'data': {'transport_mode': 'AIRPLANE', 'city': 'Atlanta', 'carrier_or_route': 'DL', 'origin': 'ATL', 'destination': 'LAX', 'hour': 12, 'day_of_week': 1, 'month': 10, 'distance_km': 3132.0, 'scheduled_duration_min': 300.0, 'historical_delay_mean': 11.2, 'temperature_c': 15.0, 'weather_condition': 'storm'}}
]

for tc in test_cases:
    res = p.predict(tc['data'])
    print(f"\n=== {tc['name']} ===")
    print(f"  Models:          Classifier: {res['classifier_name']} | Regressor: {res['regressor_name']}")
    print(f"  Delay:           Expected: {res['estimated_delay_min']}m | Typical (P50): {res['typical_median_delay_min']}m | High-End (P90): {res['high_end_p90_delay_min']}m")
    print(f"  Category Tier:   {res['predicted_category']} (Risk: {res['risk_level']})")
    print(f"  Probabilities:   {res['probabilities']} (Base Rates: {res['base_rates']})")
    print(f"  Disagreement:    {res['has_disagreement']} | Disclosure: {res['disagreement_disclosure']}")
    print("  Local Factors:")
    for f in res['important_factors']:
        print(f"    - {f['factor']}: {f['value']} [{f['impact']}]")
