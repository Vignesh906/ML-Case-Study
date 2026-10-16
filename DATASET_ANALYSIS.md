# Multimodal Transportation Delay Prediction System: Dataset Analysis

## 1. Executive Summary

This document presents a comprehensive evaluation of the transit and aviation datasets available for the **Multimodal Transportation Delay Prediction System**. The objective is to standardize and predict delay severity across four distinct transportation modalities:
- **Bus**
- **Train / Rail**
- **Metro**
- **Airplane**

Target severity categories are defined as:
- **0 = ON TIME** (0 to 5 minutes delay)
- **1 = MODERATELY DELAYED** (6 to 20 minutes delay)
- **2 = SEVERELY DELAYED** (More than 20 minutes delay)

---

## 2. Dataset Inventory & Structural Inspection

### 2.1 Overview Table

| Directory | File Name | Size (MB) | Rows | Columns | Supported Modes | Retained / Rejected | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `archive/` | `flights.csv` | 564.96 MB | 5,819,079 | 31 | Airplane | **Selected** | Primary dataset for aviation delay prediction (US DOT 2015). |
| `archive/` | `airlines.csv` | 0.0003 MB | 14 | 2 | Airplane metadata | **Selected** | Lookup table mapping 2-letter IATA carrier codes to airline names. |
| `archive/` | `airports.csv` | 0.024 MB | 322 | 7 | Airplane metadata | **Selected** | Lookup table mapping 3-letter IATA airport codes to airport name, city, state, coords. |
| `transit/` | `train.csv` | 276.45 MB | 455,520 | 88 | Bus, Rail, Metro, Tram, LRT | **Selected** | Primary multimodal transit dataset containing Bus, Rail, Metro with 88 operational features (Jan–Oct 2020). |
| `transit/` | `val.csv` | 34.74 MB | 56,940 | 88 | Bus, Rail, Metro, Tram, LRT | **Selected** | Chronological transit validation split (Oct 19 – Nov 24, 2020). |
| `transit/` | `test (1).csv` | 34.31 MB | 56,940 | 88 | Bus, Rail, Metro, Tram, LRT | **Selected** | Chronological transit test split (Nov 24 – Dec 30, 2020). |
| `transit/` | `processed.csv` | 87.08 MB | ~140,000 | 31 | Rail only | **Rejected** | Single mode only (Rail); lacks Bus and Metro, redundant with `train.csv`. |
| `transit/` | `transit_fe.csv` | 301.02 MB | ~450,000 | 88 | Tram only | **Rejected** | Single mode only (Tram); lacks Bus, Rail, and Metro. |
| `transit/` | `transit_forecasting.csv` | 20.43 MB | ~50,000 | 23 | Tram only | **Rejected** | Tram time-series forecasting only; lacks multimodal classification targets. |
| `transit/` | `transit_anomaly_detection.csv` | 9.19 MB | ~25,000 | 13 | Not specified | **Rejected** | Lacks mode disaggregation, lacks route/operational predictors. |

---

## 3. Dataset Characteristics & Column Profiles

### 3.1 Public Transportation Dataset (`transit/train.csv`, `val.csv`, `test (1).csv`)
- **Origin / Nature**: GTFS-inspired multi-city European synthetic transit simulation covering Berlin, London, Paris, Tokyo, and Singapore. Spans the complete calendar year 2020.
- **Key Columns**:
  - Identifiers: `trip_id`, `route_id`, `city`, `origin_station`, `dest_station`.
  - Mode: `route_type` (`bus`, `rail`, `metro`, `tram`, `lrt`).
  - Target: `delay_minutes` (non-negative delay in minutes, integer).
  - Pre-trip Operational Features: `scheduled_travel_min`, `route_length_km`, `headway_min`, `vehicle_capacity`, `occupancy_rate`, `boarding_count`, `passenger_count`, `weather_condition`, `temperature_c`, `is_peak_hour`, `is_weekend`, `is_holiday`.
  - Historical Rolling Delay Features: `delay_6h_mean`, `delay_24h_mean`, `delay_7d_mean`, `delay_cumsum_4h` (strictly calculated using historical observations preceding trip dispatch).
  - Safety / Leakage Columns (Excluded): `actual_travel_min`, `service_status`, `on_time_flag`.

### 3.2 Aviation Dataset (`archive/flights.csv`)
- **Origin / Nature**: Official US Department of Transportation (DOT) Bureau of Transportation Statistics (BTS) 2015 Airline On-Time Performance dataset (5,819,079 flights across 14 major US carriers and 322 commercial airports).
- **Key Columns**:
  - Identifiers: `AIRLINE`, `FLIGHT_NUMBER`, `TAIL_NUMBER`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`.
  - Temporal: `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`, `SCHEDULED_DEPARTURE` (hhmm).
  - Operational: `SCHEDULED_TIME` (minutes), `DISTANCE` (statute miles).
  - Target: `DEPARTURE_DELAY` (minutes, negative indicates early departure, positive indicates delay).
  - Safety / Leakage Columns (Excluded): `DEPARTURE_TIME`, `ARRIVAL_TIME`, `ARRIVAL_DELAY`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ELAPSED_TIME`, `AIR_TIME`, `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`, `DIVERTED`, `CANCELLED`.

---

## 4. Target Variable Analysis & Class Imbalance

### 4.1 Target Definition
The standardized target is `delay_category`:
```
delay_category = 0  (ON TIME: 0 to 5 minutes)
delay_category = 1  (MODERATELY DELAYED: 6 to 20 minutes)
delay_category = 2  (SEVERELY DELAYED: > 20 minutes)
```

### 4.2 Class Distribution Comparison
- **Public Transit**:
  - On Time (0-5 min): ~95.6%
  - Moderately Delayed (6-20 min): ~3.7%
  - Severely Delayed (>20 min): ~0.7%
- **Commercial Aviation**:
  - On Time (0-5 min): ~59.9%
  - Moderately Delayed (6-20 min): ~15.7%
  - Severely Delayed (>20 min): ~24.3%
- **Combined Multimodal Corpus (Balanced by Mode)**:
  - On Time (0): ~77.8%
  - Moderately Delayed (1): ~9.7%
  - Severely Delayed (2): ~12.5%

### 4.3 Mitigation Strategies
To address the natural class imbalance without data leakage:
1. Stratified sampling across modes and time periods.
2. Balanced class weighting (`class_weight='balanced'`) during model fitting for Logistic Regression, Decision Tree, Random Forest, and calibrated Linear SVM.
3. Distance-weighted neighborhood voting for K-Nearest Neighbors.
4. Stratified validation and test partitioning.
5. Primary evaluation metric set to **Macro F1-Score** to ensure that minority severe delay events are weighted equally with majority on-time trips.

---

## 5. Multimodal Data Harmonization Strategy

Because public transit and commercial flights have disparate source schemas, they are mapped into a unified multimodal schema:

| Unified Feature | Description | Public Transit Mapping | Aviation Mapping |
| :--- | :--- | :--- | :--- |
| `transport_mode` | Mode Indicator | `BUS`, `TRAIN` (from `rail`), `METRO` | `AIRPLANE` |
| `hour` | Hour of departure (0-23) | `hour` | `SCHEDULED_DEPARTURE // 100` |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | `day_of_week` | `DAY_OF_WEEK - 1` |
| `month` | Month (1-12) | `month` | `MONTH` |
| `day` | Day of month (1-31) | Derived from `timestamp` | `DAY` |
| `is_weekend` | Weekend binary flag | `is_weekend` | `1` if day_of_week in [5,6] else `0` |
| `is_peak_hour` | Rush hour binary flag | `is_peak_hour` | `1` if hour in [7..9, 16..19] else `0` |
| `distance_km` | Travel distance (km) | `route_length_km` | `DISTANCE * 1.60934` |
| `scheduled_duration_min` | Scheduled trip duration (min) | `scheduled_travel_min` | `SCHEDULED_TIME` |
| `carrier_or_route` | Carrier / Route identifier | `route_id` | `AIRLINE` |
| `origin` | Origin terminal / hub | `origin_station` | `ORIGIN_AIRPORT` |
| `destination` | Destination terminal / hub | `dest_station` | `DESTINATION_AIRPORT` |
| `city` | Operating city | `city` | Mapped via `airports.csv` (`CITY`) |
| `historical_delay_mean` | Prior average delay (min) | `delay_24h_mean` | Airline historical mean departure delay |
| `headway_min` | Dispatch headway (min) | `headway_min` | `0.0` (Scheduled long-distance) |
| `vehicle_capacity` | Vehicle passenger capacity | `vehicle_capacity` | `180` (Standard narrow-body aircraft) |
| `occupancy_rate` | Capacity utilization (0.0-1.0) | `occupancy_rate` | Estimated commercial load factor (~0.82) |
| `passenger_count` | Boarded passenger count | `passenger_count` | `capacity * occupancy_rate` (~148) |
| `weather_condition` | Weather category | `weather_condition` | `'clear'` (default aviation baseline) |
| `temperature_c` | Ambient temperature (°C) | `temperature_c` | Seasonal ambient mean |
| `delay_minutes` | Continuous delay (min) | `delay_minutes` | `max(0, DEPARTURE_DELAY)` |
| `delay_category` | Target severity (0, 1, 2) | Discretized `delay_minutes` | Discretized `DEPARTURE_DELAY` |

---

## 6. Data Leakage Prevention Protocol

The following architectural guarantees are implemented:
1. **Strict Temporal Feature Isolation**: All temporal and rolling statistics reflect strictly prior events.
2. **Exclusion of Arrival Features**: No `ARRIVAL_DELAY`, `ARRIVAL_TIME`, `TAXI_IN`, or `WHEELS_ON` are utilized for departure delay classification.
3. **Exclusion of Post-Departure Metrics**: `service_status`, `on_time_flag`, and `cancellation_reason` are completely discarded from model features.
4. **Chronological Splitting**: Chronological ordering is preserved between train, validation, and test partitions.
