import urllib.request
import json

base_url = "http://127.0.0.1:5050"

# 1. Test GET /
with urllib.request.urlopen(f"{base_url}/") as resp:
    print(f"GET / -> Status: {resp.status}")

# 2. Test GET /models
with urllib.request.urlopen(f"{base_url}/models") as resp:
    print(f"GET /models -> Status: {resp.status}")

# 3. Test GET /api/weather?city=Berlin
with urllib.request.urlopen(f"{base_url}/api/weather?city=Berlin") as resp:
    data = json.loads(resp.read().decode('utf-8'))
    print(f"GET /api/weather?city=Berlin -> Status: {resp.status}, Condition: {data.get('weather_condition')}, Temp: {data.get('temperature_c')}C")

# 4. Test POST /api/predict for AIRPLANE (DL, ATL -> LAX)
req_data = {
    'transport_mode': 'AIRPLANE',
    'carrier_or_route': 'DL',
    'origin': 'ATL',
    'destination': 'LAX',
    'city': 'Atlanta',
    'hour': 14,
    'day_of_week': 2,
    'month': 10,
    'distance_km': 3132.0,
    'scheduled_duration_min': 285.0,
    'weather_condition': 'clear',
    'temperature_c': 20.0
}
req = urllib.request.Request(
    f"{base_url}/api/predict",
    data=json.dumps(req_data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req) as resp:
    pred_res = json.loads(resp.read().decode('utf-8'))
    print(f"POST /api/predict (AIRPLANE DL ATL->LAX) -> Status: {resp.status}")
    print(f"  Est Delay: {pred_res.get('estimated_delay_min')} min")
    print(f"  Raw Reg: {pred_res.get('raw_regression_min')} min")
    print(f"  P50: {pred_res.get('typical_median_delay_min')} min")
    print(f"  P90: {pred_res.get('high_end_p90_delay_min')} min")
    print(f"  Category: {pred_res.get('predicted_category')}")
    print(f"  Confidence: {pred_res.get('confidence_pct')}%")
    print(f"  Risk: {pred_res.get('risk_level')}")
    print(f"  Regressor: {pred_res.get('regressor_name')}")

# 5. Test POST /api/predict for AIRPLANE (HA, HNL -> OGG, low delay carrier)
req_data_ha = {
    'transport_mode': 'AIRPLANE',
    'carrier_or_route': 'HA',
    'origin': 'HNL',
    'destination': 'OGG',
    'city': 'Honolulu',
    'hour': 10,
    'day_of_week': 2,
    'month': 10,
    'distance_km': 160.0,
    'scheduled_duration_min': 40.0,
    'weather_condition': 'clear',
    'temperature_c': 25.0
}
req2 = urllib.request.Request(
    f"{base_url}/api/predict",
    data=json.dumps(req_data_ha).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req2) as resp:
    pred_res_ha = json.loads(resp.read().decode('utf-8'))
    print(f"\nPOST /api/predict (AIRPLANE HA HNL->OGG) -> Status: {resp.status}")
    print(f"  Est Delay: {pred_res_ha.get('estimated_delay_min')} min")
    print(f"  Raw Reg: {pred_res_ha.get('raw_regression_min')} min")
    print(f"  P50: {pred_res_ha.get('typical_median_delay_min')} min")
    print(f"  P90: {pred_res_ha.get('high_end_p90_delay_min')} min")
    print(f"  Category: {pred_res_ha.get('predicted_category')}")
    print(f"  Confidence: {pred_res_ha.get('confidence_pct')}%")
    print(f"  Risk: {pred_res_ha.get('risk_level')}")
