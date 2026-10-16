# Multimodal Transportation Delay Prediction System Using Machine Learning

An end-to-end, production-grade academic Machine Learning platform that predicts delay severity across four distinct transportation modalities: **Bus**, **Train/Rail**, **Metro**, and **Airplane**.

---

## 1. Project Overview & Problem Statement

Transportation systems are the backbone of modern urban mobility and global commerce. However, unexpected delays across public transit networks and aviation corridors cause billions of dollars in economic loss, passenger inconvenience, and compounding logistical failures. 

While existing research typically models a single modality in isolation (e.g., flight delays or bus bunching), real-world commuters and transit authorities navigate an interconnected multimodal ecosystem. This project designs, trains, evaluates, and deploys a unified **Multimodal Transportation Delay Prediction System** that standardizes disparate operational schemas into a cohesive, leakage-free feature space.

---

## 2. Project Target & Delay Discretization

The system models **Transportation Delay Severity Classification** into three project-defined tiers:

| Tier | Category | Delay Threshold |
| :---: | :--- | :--- |
| **0** | **ON TIME** | 0 to 5 minutes delay |
| **1** | **MODERATELY DELAYED** | 6 to 20 minutes delay |
| **2** | **SEVERELY DELAYED** | Greater than 20 minutes delay |

*Note: These classification thresholds are project-defined research benchmarks.*

---

## 3. Dataset Description & Inventory

The project utilizes authentic multimodal data sources:

1. **Public Transportation Dataset (`transit/`)**:
   - Multi-city GTFS-inspired simulation spanning the calendar year 2020 across Berlin, London, Paris, Tokyo, and Singapore.
   - 88 original columns capturing route lengths, passenger boarding, capacity, occupancy rates, headways, weather, and historical rolling delay averages.
   - Primary subsets utilized: `train.csv` (Jan–Oct 2020), `val.csv` (Oct–Nov 2020), `test (1).csv` (Nov–Dec 2020).
   - Rejected redundant single-mode variants: `processed.csv` (rail only), `transit_fe.csv` (tram only), `transit_forecasting.csv` (tram time-series only).

2. **Commercial Aviation Dataset (`archive/`)**:
   - Official **US Department of Transportation (DOT) Bureau of Transportation Statistics (BTS)** 2015 airline on-time performance records (5.82 million flights).
   - Supporting lookup tables: `airlines.csv` (14 carrier codes) and `airports.csv` (322 airport metadata records).

3. **Harmonized Multimodal Splits (`data/processed/`)**:
   - Balanced sampling (25% Bus, 25% Train, 25% Metro, 25% Airplane):
     - **Training Split**: 80,000 records (20,000 per mode)
     - **Validation Split**: 12,000 records (3,000 per mode)
     - **Testing Split**: 12,000 records (3,000 per mode)
     - **Total Corpus**: 104,000 records

---

## 4. Feature Engineering & Strict Leakage Prevention

The system guarantees strict temporal and causal isolation:
- **Zero Target Leakage**: Arrival delays (`ARRIVAL_DELAY`, `ARRIVAL_TIME`), taxi-in times, actual travel durations, and post-trip status flags (`service_status`, `on_time_flag`) are strictly discarded.
- **Leakage-Free Historical Features**: Historical delay averages (`historical_delay_mean`) represent prior rolling baselines preceding dispatch.
- **Cyclical Temporal Encoding**: Trigonometric sine and cosine transforms for `hour` (24h period), `day_of_week` (7d period), and `month` (12m period).
- **Operational Interactions**: Estimated commercial travel speed (`speed_kmh`), vehicle crowding index (`crowding_ratio`), logarithmic distance scaling, and risk composite indices (`delay_risk_score`).

---

## 5. Machine Learning Models & Evaluation Results

Five diverse machine learning architectures were trained on the harmonized multimodal training split and evaluated on the independent test set (12,000 samples). Due to class imbalance (89.4% On Time, 6.0% Moderate, 4.6% Severe in the balanced multimodal training split), models utilized balanced class weighting, and **Macro F1-Score** served as the primary optimization metric.

### Benchmark Results (Independent Test Partition)

| Algorithm Candidate | Accuracy | Macro Precision | Macro Recall | Macro F1 (Primary) | Weighted F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Best)** | **0.7746** | **0.3992** | **0.5357** | **0.4054** | **0.8153** | 1.50s |
| **K-Nearest Neighbors** | 0.7312 | 0.3885 | 0.4952 | 0.3980 | 0.7785 | 2.15s |
| **Decision Tree** | 0.6338 | 0.3837 | 0.5260 | 0.3631 | 0.7226 | 0.48s |
| **Logistic Regression** | 0.5998 | 0.4086 | 0.4648 | 0.3567 | 0.6983 | 0.84s |
| **Linear SVM (calibrated)** | 0.9093 | 0.3031 | 0.3333 | 0.3175 | 0.8662 | 9.41s |

**Automatic Model Selection**: **Random Forest** achieved the highest Macro F1 score (**0.4054**) and was automatically serialized to `models/best_model.joblib`.

---

## 6. System Architecture & Project Structure

```text
ML case study/
├── data/
│   ├── raw/                 # Original archive/ and transit/ datasets
│   └── processed/           # Harmonized multimodal_train.csv, val, test
├── models/
│   ├── best_model.joblib
│   ├── all_models.joblib
│   └── preprocessing_pipeline.joblib
├── results/
│   ├── metrics/             # JSON metrics and classification reports
│   ├── plots/               # Evaluation charts (confusion matrix, ROC, comparison)
│   └── reports/             # Textual training summaries
├── src/
│   ├── data_loader.py       # Ingestion & multimodal standardization
│   ├── feature_engineering.py # Cyclical & operational feature transforms
│   ├── preprocessing.py     # ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── train_models.py      # ML model training & imbalance balancing
│   ├── evaluate_models.py   # Visualization generator
│   ├── prediction.py        # Real-time inference engine
│   └── analytics_data.py    # Summary caching service
├── templates/
│   ├── base.html            # Core layout & navigation
│   ├── dashboard.html       # Executive summary & modal benchmarks
│   ├── prediction.html      # Dynamic multi-modal input & live results
│   ├── models.html          # Benchmark table & confusion matrices
│   ├── analytics.html       # Hourly trends & feature importances
│   └── about.html           # Documentation & methodology
├── static/
│   ├── css/style.css        # Modern glassmorphism CSS
│   └── js/main.js           # Dynamic mode switching & AJAX inference
├── tests/
│   └── test_system.py       # Automated unit & integration test suite
├── app.py                   # Flask web server
├── train.py                 # Master training workflow
├── requirements.txt         # Dependency declarations
├── DATASET_ANALYSIS.md      # Detailed dataset evaluation report
└── PROJECT_REPORT.md        # Comprehensive academic research report
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9 on Windows)
- pip

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Execute Master Training Workflow
To train all 5 algorithms, evaluate on test partitions, select the best model, save all artifacts, and generate visualization plots:
```bash
python train.py
```

### Step 3: Run Automated Test Suite
```bash
python -m unittest tests/test_system.py
```

### Step 4: Launch Flask Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5050
```

---

## 8. Web Dashboard Features

1. **Executive Dashboard (`/`)**: Total records counter, supported modes breakdown, best model indicator, target delay severity donut chart, and cross-modality comparison charts.
2. **Interactive Prediction Interface (`/prediction`)**: Dynamically adapts input form based on active modality (Bus, Train, Metro, Airplane), invokes model inference asynchronously, and renders severity badges, calibrated probabilities, and Explainable AI contributing factors.
3. **Model Benchmark Page (`/models`)**: Detailed metric comparison table across all 5 models, best model highlight, grouped bar chart, and interactive confusion matrix heatmap viewer.
4. **Operational Analytics (`/analytics`)**: Diurnal congestion curves across 24 hours, day-of-week trends, average delay duration by mode, and top feature importances.
5. **Academic Documentation (`/about`)**: Architecture flowcharts, dataset characteristics, and leakage prevention protocol.

---

## 9. Important Limitations

1. **Synthetic Transit Telemetry**: The public transit dataset exhibits GTFS-inspired simulation characteristics across 5 global cities rather than real-world physical GPS pings.
2. **Aviation Historical Period**: Aviation records reflect the 2015 calendar year from the US Department of Transportation.
3. **Threshold Notice**: Delay severity thresholds (0-5 min, 6-20 min, >20 min) are project-defined academic benchmarks.

---

## 10. Future Enhancements

- Integration of live General Transit Feed Specification (GTFS-RT) feeds.
- Real-time ADS-B flight telemetry and FAA National Airspace System (NAS) ground delay programs.
- Deep Learning sequence architectures (LSTM / Temporal Fusion Transformers) for continuous time-series forecasting.
- Geospatial weather radar API integration (NOAA / OpenWeatherMap).

---

## 11. Technologies Used

- **Programming Language**: Python 3.11
- **Machine Learning**: Scikit-Learn, Joblib, NumPy, Pandas, SciPy
- **Web Backend**: Flask, Jinja2, Werkzeug
- **Frontend / Visualizations**: Vanilla CSS (Glassmorphism), Vanilla JavaScript, Plotly.js, Matplotlib, Seaborn
