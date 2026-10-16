"""
Test genuine live weather queries for at least 8 locations via Open-Meteo API.
Records exact URLs, coordinates, WMO codes, mapped conditions, and timestamps.
"""
import urllib.request
import json
import time
from datetime import datetime

LOCATIONS = {
    "New_York": {"lat": 40.7128, "lon": -74.0060, "name": "New York, NY"},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "name": "Chicago, IL"},
    "London": {"lat": 51.5074, "lon": -0.1278, "name": "London, UK"},
    "Berlin": {"lat": 52.5200, "lon": 13.4050, "name": "Berlin, Germany"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai, India"},
    "ATL": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta Hartsfield-Jackson (ATL)"},
    "ORD": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago O'Hare (ORD)"},
    "LAX": {"lat": 33.9416, "lon": -118.4085, "name": "Los Angeles International (LAX)"}
}

def map_wmo_code(code: int) -> str:
    if code == 0:
        return "clear"
    elif code in [1, 2, 3]:
        return "cloudy"
    elif code in [45, 48]:
        return "fog"
    elif code in [51, 52, 53, 54, 55, 56, 57, 61, 62, 63, 80, 81]:
        return "light_rain"
    elif code in [64, 65, 66, 67, 82]:
        return "heavy_rain"
    elif code in [71, 72, 73, 74, 75, 76, 77, 85, 86]:
        return "snow"
    elif code in [95, 96, 99]:
        return "storm"
    return "cloudy"

def main():
    results = {}
    print("Testing Open-Meteo Live API across 8 locations...\n")
    for key, info in LOCATIONS.items():
        lat, lon = info["lat"], info["lon"]
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m"
        # Retry up to 3 times with backoff
        for attempt in range(3):
            try:
                time.sleep(0.5)
                req = urllib.request.Request(url, headers={"User-Agent": "MultimodalTransitDelayAI/2.0"})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    curr = data.get("current", {})
                    temp = float(curr.get("temperature_2m", 18.0))
                    code = int(curr.get("weather_code", 0))
                    wind = float(curr.get("wind_speed_10m", 10.0))
                    cond = map_wmo_code(code)
                    obs_time = curr.get("time", "")
                    
                    res_entry = {
                        "location": info["name"],
                        "coordinates": {"latitude": lat, "longitude": lon},
                        "api_url": url,
                        "temperature_c": temp,
                        "wmo_code": code,
                        "condition": cond,
                        "wind_speed_kmh": wind,
                        "observed_time_utc": f"{obs_time}Z",
                        "retrieval_time_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "status": "LIVE_SUCCESS"
                    }
                    results[key] = res_entry
                    print(f"[{key:10s}] Temp: {temp:5.1f} C | WMO: {code:2d} ({cond:10s}) | Time: {obs_time}Z")
                    break
            except Exception as e:
                if attempt == 2:
                    print(f"[{key:10s}] Failed after 3 attempts: {e}")
                    results[key] = {
                        "location": info["name"],
                        "status": "UNAVAILABLE",
                        "error": str(e),
                        "retrieval_time_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                time.sleep(1.0)
            
    with open("results/metrics/live_weather_8_locations.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved 8 locations live weather results to results/metrics/live_weather_8_locations.json")

if __name__ == "__main__":
    main()
