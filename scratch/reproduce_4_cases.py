import sys, os
sys.path.insert(0, os.path.abspath('.'))
from src.prediction import DelayPredictor
import json

p = DelayPredictor()

cases = [
    ('Bus London', {
        'transport_mode': 'BUS', 'city': 'London', 'carrier_or_route': 'RT_LON_001',
        'origin': 'STN_LON_0037', 'destination': 'STN_LON_0211',
        'hour': 8, 'day_of_week': 1, 'month': 10, 'weather_condition': 'cloudy',
        'temperature_c': 17.8, 'occupancy_rate': 0.65, 'historical_delay_mean': 5.2,
        'headway_min': 15
    }),
    ('Train Berlin', {
        'transport_mode': 'TRAIN', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_002',
        'origin': 'STN_BER_0001', 'destination': 'STN_BER_0010',
        'hour': 8, 'day_of_week': 1, 'month': 10, 'weather_condition': 'cloudy',
        'temperature_c': 11.1, 'historical_delay_mean': 4.8, 'headway_min': 48
    }),
    ('Metro Berlin', {
        'transport_mode': 'METRO', 'city': 'Berlin', 'carrier_or_route': 'RT_BER_001',
        'origin': 'STN_BER_0001', 'destination': 'STN_BER_0005',
        'hour': 8, 'day_of_week': 1, 'month': 10, 'weather_condition': 'cloudy',
        'temperature_c': 11.1, 'historical_delay_mean': 6.2
    }),
    ('Airplane DL ATL->LAX', {
        'transport_mode': 'AIRPLANE', 'carrier_or_route': 'DL', 'origin': 'ATL',
        'destination': 'LAX', 'hour': 8, 'day_of_week': 1, 'month': 10,
        'weather_condition': 'clear', 'temperature_c': 17.4
    })
]

print('=== CURRENT PREDICTION PATH OUTPUTS ===')
for name, payload in cases:
    res = p.predict(payload)
    probs = res['probabilities']
    exp_d = res['expected_delay_min']
    p90 = res['high_end_risk_p90_min']
    p_any = res.get('chance_any_delay_pct')
    p_gt5 = res.get('chance_gt5_pct')
    tier = res['predicted_category']
    risk = res.get('risk_level')
    banner = res.get('disagreement_disclosure')
    p_mod = probs['MODERATELY DELAYED'] / 100.0
    p_sev = probs['SEVERELY DELAYED'] / 100.0
    floor_check = 5.0 * p_mod + 20.0 * p_sev
    
    print(f'*** {name} ***')
    print(f'  Probabilities: {probs}')
    print(f'  Expected Delay: {exp_d} min (Floor 5*P(Mod)+20*P(Sev) = {floor_check:.2f}) -> Violates: {exp_d < floor_check - 0.01}')
    p90_violates = (p_sev > 0.10 and p90 <= 20.0)
    print(f'  P90 Delay: {p90} min -> P(Sev)={probs["SEVERELY DELAYED"]}%, Violates P90>20 when P(Sev)>10%: {p90_violates}')
    print(f'  Chance of any delay: {p_any}%')
    print(f'  Chance of >5 min: {p_gt5}%')
    print(f'  Tier: {tier} | Risk: {risk}')
    print(f'  Disagreement Banner: {banner}')
    print()
