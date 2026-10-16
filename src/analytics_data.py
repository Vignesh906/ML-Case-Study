"""
Analytics and Dataset Summary Cache
------------------------------------
Precomputes and provides cached summary statistics for the web dashboard,
including record counts, modal breakdowns, delay category frequencies,
hourly patterns, and weather impacts.
"""

import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
STATS_CACHE_FILE = os.path.join(BASE_DIR, "results", "metrics", "dataset_stats.json")

CLASS_NAMES = ["ON TIME", "MODERATELY DELAYED", "SEVERELY DELAYED"]

def compute_dataset_stats():
    train_path = os.path.join(PROCESSED_DIR, "multimodal_train.csv")
    val_path = os.path.join(PROCESSED_DIR, "multimodal_val.csv")
    test_path = os.path.join(PROCESSED_DIR, "multimodal_test.csv")

    if not os.path.exists(train_path):
        return {}

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()
    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()

    total_records = len(df_train) + len(df_val) + len(df_test)
    modes = df_train['transport_mode'].value_counts().to_dict()

    # Delay distribution across train
    delay_dist_raw = df_train['delay_category'].value_counts().to_dict()
    delay_dist = {
        CLASS_NAMES[k]: {
            "count": int(v),
            "pct": round(float(v) / len(df_train) * 100, 1)
        }
        for k, v in delay_dist_raw.items()
    }

    # Delay rate by mode
    mode_stats = {}
    for m in ['BUS', 'TRAIN', 'METRO', 'AIRPLANE']:
        sub = df_train[df_train['transport_mode'] == m]
        if len(sub) > 0:
            on_time_pct = (sub['delay_category'] == 0).mean() * 100
            mod_pct = (sub['delay_category'] == 1).mean() * 100
            sev_pct = (sub['delay_category'] == 2).mean() * 100
            avg_delay = sub['delay_minutes'].mean()
            mode_stats[m] = {
                "records": len(sub),
                "avg_delay_min": round(float(avg_delay), 1),
                "on_time_pct": round(float(on_time_pct), 1),
                "moderate_delay_pct": round(float(mod_pct), 1),
                "severe_delay_pct": round(float(sev_pct), 1)
            }

    # Hourly delay trend
    hourly_trend = []
    for h in range(24):
        sub_h = df_train[df_train['hour'] == h]
        delayed_pct = (sub_h['delay_category'] > 0).mean() * 100 if len(sub_h) > 0 else 0.0
        hourly_trend.append({"hour": h, "delay_pct": round(float(delayed_pct), 1)})

    # Day of week trend
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_trend = []
    for d in range(7):
        sub_d = df_train[df_train['day_of_week'] == d]
        delayed_pct = (sub_d['delay_category'] > 0).mean() * 100 if len(sub_d) > 0 else 0.0
        dow_trend.append({"day": dow_names[d], "delay_pct": round(float(delayed_pct), 1)})

    stats = {
        "total_records": total_records,
        "train_records": len(df_train),
        "val_records": len(df_val),
        "test_records": len(df_test),
        "feature_count": len(df_train.columns) - 2,
        "modes": modes,
        "delay_distribution": delay_dist,
        "mode_stats": mode_stats,
        "hourly_trend": hourly_trend,
        "dow_trend": dow_trend
    }

    os.makedirs(os.path.dirname(STATS_CACHE_FILE), exist_ok=True)
    with open(STATS_CACHE_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    return stats

def get_dataset_stats():
    if os.path.exists(STATS_CACHE_FILE):
        with open(STATS_CACHE_FILE, "r") as f:
            return json.load(f)
    return compute_dataset_stats()

if __name__ == "__main__":
    s = compute_dataset_stats()
    print("Computed dataset stats:", s.keys())
