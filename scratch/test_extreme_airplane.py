import urllib.request
import json

base_url = "http://127.0.0.1:5050"

# Extreme condition test
req_data_extreme = {
    'transport_mode': 'AIRPLANE',
    'carrier_or_route': 'UA',
    'origin': 'ORD',
    'destination': 'DEN',
    'city': 'Chicago',
    'hour': 18,
    'day_of_week': 4,
    'month': 10,
    'distance_km': 1450.0,
    'scheduled_duration_min': 160.0,
    'historical_delay_mean': 30.0,
    'weather_condition': 'storm',
    'temperature_c': -5.0
}
req3 = urllib.request.Request(
    f"{base_url}/api/predict",
    data=json.dumps(req_data_extreme).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req3) as resp:
    pred_res_ext = json.loads(resp.read().decode('utf-8'))
    print(f"\nPOST /api/predict (AIRPLANE UA ORD->DEN Storm Extreme) -> Status: {resp.status}")
    print(f"  Est Delay: {pred_res_ext.get('estimated_delay_min')} min")
    print(f"  Raw Reg: {pred_res_ext.get('raw_regression_min')} min")
    print(f"  P50: {pred_res_ext.get('typical_median_delay_min')} min")
    print(f"  P90: {pred_res_ext.get('high_end_p90_delay_min')} min")
    print(f"  Category: {pred_res_ext.get('predicted_category')}")
    print(f"  Confidence: {pred_res_ext.get('confidence_pct')}%")
    print(f"  Risk: {pred_res_ext.get('risk_level')}")
