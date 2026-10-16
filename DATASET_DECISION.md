# Dataset Decision: Academic Scope vs. Operational Reality

**Document:** `DATASET_DECISION.md`  
**Audience:** Academic Evaluators, Data Science Reviewers & Software Engineers  
**Project:** Multimodal Transportation Delay Prediction System

---

## 1. Core Question: Are the Existing Datasets Sufficient?

**Yes.** For an advanced undergraduate / master's-level Machine Learning Case Study, the existing datasets are **fully sufficient, statistically sound, and methodologically appropriate.**

Replacing the datasets with arbitrary external downloads is **not recommended** for the reasons outlined below.

---

## 2. Academic Case Study vs. Operational Deployment

Understanding the distinction between an academic machine learning case study and a live industrial dispatch system is critical for methodological rigor:

```
+-----------------------------------------------------------------------------------+
|                           ACADEMIC ML CASE STUDY (OUR FOCUS)                      |
|  - Consistent, leakage-free multimodal schema across 4 modes                      |
|  - Valid chronological train/val/test splits                                      |
|  - Benchmarked against 5 classical & ensemble ML algorithms                       |
|  - Proper handling of severe multiclass imbalance (89% on-time vs 11% delayed)    |
|  - Explainable feature attributions & calibrated class probabilities              |
+-----------------------------------------------------------------------------------+
                                         vs.
+-----------------------------------------------------------------------------------+
|                        COMMERCIAL REAL-TIME DISPATCH ENGINE                       |
|  - Requires live vehicle GPS pings (GTFS-Realtime Protobuf feeds every 15s)       |
|  - Requires live FAA radar telemetry & ATC ground-delay program feeds             |
|  - Massive operational cloud infrastructure (Kafka, Apache Flink, Redis)          |
|  - Commercial API keys & proprietary telemetry licensing                          |
+-----------------------------------------------------------------------------------+
```

### Why the Existing Datasets Excel Academically:
1. **Size & Depth:** Over **6.27 million combined records** (5.82M aviation rows, 455k transit rows).
2. **Leakage-Free Temporal Splits:** The dataset features strict chronological partitioning (Train: Jan–Oct 2020 / Jan–Sep 2015, Val: Oct / Oct, Test: Nov–Dec / Nov–Dec).
3. **Harmonized Multimodal Schema:** Bridges public road transit, heavy rail, underground metro, and commercial aviation into a single 39-feature space.
4. **Reproducibility:** Evaluators and students can reproduce all training, validation, and testing steps locally in minutes without paid cloud API dependencies.

---

## 3. Evaluation of Candidate External Datasets

We investigated whether supplementing or replacing the datasets with external sources would genuinely improve the project:

| Candidate Source | Evaluation | Verdict |
| :--- | :--- | :--- |
| **Kaggle UK Train Punctuality (Network Rail)** | High-volume raw timetable records, but lacks weather, vehicle capacity, and multimodal compatibility. Merging it would break cross-mode harmonization. | **Do Not Adopt** |
| **Transitland / TfL Unified API** | Requires live API keys and per-city authentication. Transit agencies do not provide historical 2020 delay records in clean static tabular CSV format. | **Do Not Adopt** |
| **Open-Meteo Weather API** | Free, open-access, keyless live meteorological telemetry for any latitude/longitude coordinates globally. Directly maps to `temperature_c` and `weather_condition`. | **Adopt as Live Environmental Telemetry** |

---

## 4. Final Architectural Decision

1. **Retain Existing Datasets:**
   - Keep `transit/train.csv` (Bus, Train, Metro) and `archive/flights.csv` (Airplane).
   - Downsample into clean 80k/12k/12k balanced multimodal splits for fast, reliable local training.

2. **Fix Dropdown Data Generation:**
   - Enhance `src/generate_routes_metadata.py` and `static/js/main.js` so that transit dropdowns display all valid terminal stops served on each line, rather than constraining each route to a single origin/destination.

3. **Enhance Feature Engineering & Model Balancing:**
   - Incorporate mode interaction features and class-balanced sample weights into `train_models.py`.
   - Report overall and mode-specific performance on the held-out test set.

4. **Honest & Transparent System Labeling:**
   - Accurately describe the platform as:  
     *"Multimodal Transportation Delay Prediction System — Calibrated Machine Learning Pipeline Backed by Empirical US DOT and Urban Transit Datasets."*
   - Integrate optional live weather telemetry via **Open-Meteo** to provide real-time ambient conditions without fabricating operational delay data.
