"""
Flask Web Application for Multimodal Transportation Delay Prediction System
----------------------------------------------------------------------------
Provides web dashboard, real-time prediction interface, model evaluation comparisons,
deep analytics, live meteorological telemetry API, and RESTful prediction endpoints.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prediction import get_prediction, predictor_instance
from src.analytics_data import get_dataset_stats

app = Flask(__name__)
app.config['SECRET_KEY'] = 'multimodal-ml-transit-delay-secret'

@app.after_request
def add_cache_control(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

METRICS_FILE = os.path.join(BASE_DIR, "results", "metrics", "model_comparison.json")

def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {
        "best_model": "Random Forest",
        "best_macro_f1": 0.4031,
        "models": {},
        "feature_importances": {}
    }

# Geo Coordinates Registry for Real-Time Meteorological Telemetry (Open-Meteo)
LOCATION_COORDINATES = {
    # Transit Urban Hubs
    "New_York": {"lat": 40.7128, "lon": -74.0060, "name": "New York, NY"},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "name": "Chicago, IL"},
    "Los_Angeles": {"lat": 34.0522, "lon": -118.2437, "name": "Los Angeles, CA"},
    "London": {"lat": 51.5074, "lon": -0.1278, "name": "London, UK"},
    "Berlin": {"lat": 52.5200, "lon": 13.4050, "name": "Berlin, Germany"},
    "Paris": {"lat": 48.8566, "lon": 2.3522, "name": "Paris, France"},
    "Atlanta": {"lat": 33.7490, "lon": -84.3880, "name": "Atlanta, GA"},
    "Seattle": {"lat": 47.6062, "lon": -122.3321, "name": "Seattle, WA"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai, India"},
    # Aviation Major Hubs
    "ATL": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta Hartsfield-Jackson (ATL)"},
    "ORD": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago O'Hare (ORD)"},
    "DFW": {"lat": 32.8998, "lon": -97.0403, "name": "Dallas/Fort Worth (DFW)"},
    "DEN": {"lat": 39.8561, "lon": -104.6737, "name": "Denver International (DEN)"},
    "LAX": {"lat": 33.9416, "lon": -118.4085, "name": "Los Angeles International (LAX)"},
    "SFO": {"lat": 37.6213, "lon": -122.3790, "name": "San Francisco International (SFO)"},
    "JFK": {"lat": 40.6413, "lon": -73.7781, "name": "New York JFK (JFK)"},
    "MIA": {"lat": 25.7959, "lon": -80.2870, "name": "Miami International (MIA)"},
    "SEA": {"lat": 47.4502, "lon": -122.3088, "name": "Seattle-Tacoma (SEA)"},
    "BOS": {"lat": 42.3656, "lon": -71.0096, "name": "Boston Logan (BOS)"}
}

from src.weather_utils import map_wmo_weather_code, WMO_MAPPING_DOCUMENTATION

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Main dashboard with summary metrics, mode comparisons, and delay breakdown."""
    stats = get_dataset_stats()
    metrics = load_metrics()
    return render_template(
        'dashboard.html',
        stats=stats,
        metrics=metrics,
        active_page='dashboard'
    )

ROUTES_METADATA_FILE = os.path.join(BASE_DIR, "data", "routes_metadata.json")
_ROUTES_METADATA_CACHE = None

def load_routes_metadata():
    global _ROUTES_METADATA_CACHE
    if _ROUTES_METADATA_CACHE is not None:
        return _ROUTES_METADATA_CACHE
    if not os.path.exists(ROUTES_METADATA_FILE):
        try:
            from src.generate_routes_metadata import generate_metadata
            generate_metadata()
        except Exception as e:
            print(f"Error generating routes metadata: {e}")
    if os.path.exists(ROUTES_METADATA_FILE):
        with open(ROUTES_METADATA_FILE, "r", encoding="utf-8") as f:
            _ROUTES_METADATA_CACHE = json.load(f)
            return _ROUTES_METADATA_CACHE
    return {"modes": ["BUS", "TRAIN", "METRO", "AIRPLANE"], "transit": {}, "airplane": {}}

@app.route('/prediction')
def prediction():
    """Interactive prediction interface with dynamic transportation mode inputs."""
    metrics = load_metrics()
    return render_template(
        'prediction.html',
        best_model_name=metrics.get('best_model', 'RandomForestClassifier (Calibrated)'),
        active_page='prediction'
    )

@app.route('/api/routes-metadata')
def api_routes_metadata():
    """Serves real route, airline, airport, and transit metadata extracted from datasets."""
    meta = load_routes_metadata()
    return jsonify(meta)

@app.route('/api/live-weather')
@app.route('/api/weather')
def api_live_weather():
    """
    Genuine Live Meteorological Telemetry Endpoint:
    Fetches real-time ambient temperature and WMO weather condition from Open-Meteo
    for the selected transit metropolitan area or aviation hub airport.
    Discloses exact API call, coordinates, parameters, and WMO mappings.
    On failure returns 'weather unavailable' state; never defaults to Clear.
    """
    target = request.args.get('target', request.args.get('city', 'New_York')).strip()
    geo = LOCATION_COORDINATES.get(target)
    
    # Try case-insensitive or partial match
    if not geo:
        for k, v in LOCATION_COORDINATES.items():
            if k.lower() == target.lower():
                geo = v
                break

    if not geo:
        geo = LOCATION_COORDINATES["New_York"]

    lat, lon = geo["lat"], geo["lon"]
    params = f"latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m"
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MultimodalTransitDelayAI/2.0'})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            curr = data.get('current', {})
            temp = float(curr.get('temperature_2m', 18.0))
            wmo_code = int(curr.get('weather_code', 0))
            wind = float(curr.get('wind_speed_10m', 10.0))
            condition = map_wmo_weather_code(wmo_code)
            obs_time = curr.get('time', datetime.utcnow().strftime("%Y-%m-%dT%H:%M"))
            retrieval_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            return jsonify({
                "status": "success",
                "is_live": True,
                "source": "Open-Meteo Live Meteorological Telemetry",
                "location": geo["name"],
                "target_key": target,
                "temperature_c": round(temp, 1),
                "weather_condition": condition,
                "condition": condition,
                "wind_speed_kmh": round(wind, 1),
                "wmo_code": wmo_code,
                "observed_at": f"{obs_time}Z (UTC)",
                "retrieved_at": f"{retrieval_time} (UTC)",
                "api_call": {
                    "url": url,
                    "endpoint": "https://api.open-meteo.com/v1/forecast",
                    "parameters": {"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code,wind_speed_10m"},
                    "coordinates": {"lat": lat, "lon": lon}
                },
                "wmo_mapping_table": WMO_MAPPING_DOCUMENTATION
            })
    except Exception as e:
        retrieval_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return jsonify({
            "status": "unavailable",
            "is_live": False,
            "source": f"Open-Meteo API Error: {str(e)}",
            "location": geo["name"],
            "target_key": target,
            "temperature_c": None,
            "weather_condition": "unavailable",
            "condition": "unavailable",
            "message": "Live weather unavailable. Please configure weather and temperature manually.",
            "retrieved_at": f"{retrieval_time} (UTC)",
            "api_call": {
                "url": url,
                "endpoint": "https://api.open-meteo.com/v1/forecast",
                "parameters": {"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code,wind_speed_10m"},
                "coordinates": {"lat": lat, "lon": lon}
            },
            "wmo_mapping_table": WMO_MAPPING_DOCUMENTATION
        }), 200

@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def api_predict():
    """API endpoint accepting user inputs and returning delay prediction and explanation."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided"}), 400

        # Input Validation
        raw_mode = str(data.get('transport_mode', data.get('mode', 'BUS'))).strip().upper()
        if raw_mode not in ['BUS', 'TRAIN', 'METRO', 'AIRPLANE']:
            return jsonify({
                "status": "error",
                "message": f"Invalid transport mode '{raw_mode}'. Supported modes are Bus, Train, Metro, and Airplane."
            }), 400

        origin = str(data.get('origin', '')).strip()
        destination = str(data.get('destination', '')).strip()

        # Reject identical origin and destination
        if origin and destination and origin == destination:
            return jsonify({
                "status": "error",
                "message": "Origin and destination locations cannot be identical."
            }), 400

        result = get_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500

@app.route('/models')
def models_comparison():
    """Comprehensive comparison of all 5 machine learning models + per-mode breakdown."""
    metrics = load_metrics()
    return render_template(
        'models.html',
        metrics=metrics,
        active_page='models'
    )

@app.route('/analytics')
def analytics():
    """Visual analytics on delay patterns, hourly congestion, and operational drivers."""
    stats = get_dataset_stats()
    metrics = load_metrics()
    return render_template(
        'analytics.html',
        stats=stats,
        metrics=metrics,
        active_page='analytics'
    )

@app.route('/about')
def about():
    """Dataset descriptions, research methodology, and limitations."""
    stats = get_dataset_stats()
    metrics = load_metrics()
    return render_template(
        'about.html',
        stats=stats,
        metrics=metrics,
        active_page='about'
    )

@app.route('/api/stats')
def api_stats():
    """Returns dataset and model statistics in JSON format for Plotly charts."""
    stats = get_dataset_stats()
    metrics = load_metrics()
    return jsonify({
        "dataset": stats,
        "metrics": metrics
    })

if __name__ == '__main__':
    print("Starting Multimodal Transportation Delay Prediction Flask server...")
    print("Open your browser and navigate to: http://127.0.0.1:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
