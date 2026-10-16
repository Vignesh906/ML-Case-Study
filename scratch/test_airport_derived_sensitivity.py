import os
import sys

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prediction import DelayPredictor

p = DelayPredictor()

test_airports = ['SEA', 'MSP', 'ATL', 'LAX', 'SFO', 'JFK', 'ORD']
print(f"{'Airport':<10} | {'Carrier':<8} | {'Est Delay':<10} | {'Raw Reg':<10} | {'P90':<8} | {'Tier':<16} | {'P(OnTime)':<10}")
print("-" * 80)

for ap in test_airports:
    inp = {
        'transport_mode': 'AIRPLANE',
        'carrier_or_route': 'DL',
        'origin': ap,
        'destination': 'LAX' if ap != 'LAX' else 'ATL',
        'hour': 14,
        'day_of_week': 2,
        'month': 10,
        'day': 14,
        'weather_condition': 'clear',
        'temperature_c': 20.0
    }
    res = p.predict(inp)
    est = res['estimated_delay_min']
    raw = res['raw_regression_min']
    p90 = res['high_end_p90_delay_min']
    tier = res['predicted_category']
    p_on = res['probabilities']['ON TIME']
    print(f"{ap:<10} | {'DL':<8} | {est:<10} | {raw:<10.4f} | {str(p90):<8} | {tier:<16} | {p_on:<10.1f}")
