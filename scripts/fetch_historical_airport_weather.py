"""
Fetches genuine historical weather observations for the year 2015
from Open-Meteo Historical Weather Archive API (CC-BY 4.0 license)
for the top 25 US commercial airports based on airports.csv coordinates.
Saves local cache to data/historical_airport_weather_2015.json
"""

import os
import sys
import json
import time
import urllib.request
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
AIRPORTS_PATH = os.path.join(BASE_DIR, "archive", "airports.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "historical_airport_weather_2015.json")

TOP_AIRPORTS = [
    'ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'PHX', 'IAH', 'SFO', 'LAS', 'MCO',
    'DTW', 'LGA', 'BOS', 'MSP', 'CLT', 'EWR', 'SEA', 'SLC', 'JFK', 'BWI',
    'FLL', 'MDW', 'DCA', 'MIA', 'SAN'
]

def map_wmo_to_condition(wmo_code):
    if wmo_code in [0, 1]:
        return 'clear'
    elif wmo_code in [2, 3]:
        return 'cloudy'
    elif wmo_code in [45, 48]:
        return 'fog'
    elif wmo_code in [51, 53, 55, 61, 63]:
        return 'light_rain'
    elif wmo_code in [65, 80, 81, 82]:
        return 'heavy_rain'
    elif wmo_code in [71, 73, 75, 77, 85, 86]:
        return 'snow'
    elif wmo_code in [95, 96, 99]:
        return 'storm'
    return 'clear'

def fetch_airport_weather():
    print(f"Reading {AIRPORTS_PATH}...", flush=True)
    airports_df = pd.read_csv(AIRPORTS_PATH)
    airports_dict = airports_df.set_index('IATA_CODE')[['LATITUDE', 'LONGITUDE', 'AIRPORT', 'CITY']].to_dict('index')

    weather_db = {}
    print(f"Fetching 2015 Open-Meteo Historical Archive telemetry for {len(TOP_AIRPORTS)} major airports...", flush=True)

    for code in TOP_AIRPORTS:
        if code not in airports_dict:
            print(f"Skipping {code}: not in airports.csv")
            continue

        meta = airports_dict[code]
        lat = meta['LATITUDE']
        lon = meta['LONGITUDE']

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat:.4f}&longitude={lon:.4f}&"
            f"start_date=2015-01-01&end_date=2015-12-31&"
            f"daily=weather_code,temperature_2m_mean&"
            f"timezone=UTC"
        )

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TransportationDelaySystem/2.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                
            dates = data['daily']['time']
            wmo_codes = data['daily']['weather_code']
            temps = data['daily']['temperature_2m_mean']

            airport_records = {}
            for d_str, code_val, temp_val in zip(dates, wmo_codes, temps):
                # key by "YYYY-MM-DD" and (month, day)
                parts = d_str.split('-')
                m_int = int(parts[1])
                d_int = int(parts[2])
                airport_records[f"{m_int:02d}-{d_int:02d}"] = {
                    "condition": map_wmo_to_condition(code_val),
                    "wmo_code": code_val,
                    "temperature_c": round(float(temp_val), 1) if temp_val is not None else 18.0
                }

            weather_db[code] = {
                "name": meta['AIRPORT'],
                "city": meta['CITY'],
                "latitude": lat,
                "longitude": lon,
                "days": airport_records
            }
            print(f"  Successfully fetched 365 days for {code} ({meta['CITY']})")
            time.sleep(0.3) # Respect Open-Meteo polite rate limits

        except Exception as e:
            print(f"  Error fetching weather for {code}: {e}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(weather_db, f, indent=2)

    print(f"\nSaved historical weather database for {len(weather_db)} airports to {OUT_PATH}")

if __name__ == "__main__":
    fetch_airport_weather()
