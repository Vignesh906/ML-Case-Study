"""
Weather Utilities and WMO Standard Interpretation Mapping
=========================================================
Strictly aligns live weather telemetry with training dataset preprocessing.
WMO Code Standards:
- 0: clear (Clear sky)
- 1, 2, 3: cloudy (Mainly clear, partly cloudy, overcast)
- 45, 48: fog (Fog, depositing rime fog)
- 51-57: light_rain (Drizzle: light, moderate, dense)
- 61-63: light_rain (Rain: slight, moderate)
- 64-67: heavy_rain (Heavy rain, freezing rain)
- 71-77, 85, 86: snow (Snow fall, snow grains, snow showers)
- 80, 81: light_rain (Rain showers)
- 82: heavy_rain (Violent rain showers)
- 95, 96, 99: storm (Thunderstorms)
"""

def map_wmo_weather_code(code: int) -> str:
    """Standardized WMO Weather code to 7 categorical features."""
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

WMO_MAPPING_DOCUMENTATION = {
    "0": "clear (Clear sky)",
    "1-3": "cloudy (Partly cloudy, overcast)",
    "45, 48": "fog (Fog, rime fog)",
    "51-57": "light_rain (Drizzle: light, moderate, dense)",
    "61-63": "light_rain (Rain: slight, moderate)",
    "64-67": "heavy_rain (Heavy rain, freezing rain)",
    "71-77": "snow (Snow fall, snow grains)",
    "80-81": "light_rain (Rain showers)",
    "82": "heavy_rain (Violent rain showers)",
    "95-99": "storm (Thunderstorm, hail)"
}
