import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"

def audit_aviation_target():
    flights_path = os.path.join(BASE_DIR, "archive", "flights.csv")
    print(f"Reading sample of flights.csv from {flights_path}...")
    
    # Read a sample of 200,000 rows
    df_raw = pd.read_csv(flights_path, nrows=200000)
    print(f"Total sampled rows: {len(df_raw)}")
    print(f"Columns: {list(df_raw.columns)}")
    
    dep_delay = df_raw['DEPARTURE_DELAY']
    arr_delay = df_raw['ARRIVAL_DELAY']
    
    print("\n=== RAW DEPARTURE_DELAY in flights.csv ===")
    print(f"Missing count: {dep_delay.isnull().sum()} ({dep_delay.isnull().mean()*100:.2f}%)")
    valid_dep = dep_delay.dropna()
    print(f"Valid count: {len(valid_dep)}")
    print(f"Mean: {valid_dep.mean():.4f}")
    print(f"Median: {valid_dep.median():.4f}")
    print(f"Std: {valid_dep.std():.4f}")
    print(f"Min: {valid_dep.min():.4f}, Max: {valid_dep.max():.4f}")
    print(f"Negative count (< 0): {(valid_dep < 0).sum()} ({(valid_dep < 0).mean()*100:.2f}%)")
    print(f"Zero count (== 0): {(valid_dep == 0).sum()} ({(valid_dep == 0).mean()*100:.2f}%)")
    print(f"<= 5 min count: {(valid_dep <= 5).sum()} ({(valid_dep <= 5).mean()*100:.2f}%)")
    print(f"5 to 10 min count: {((valid_dep > 5) & (valid_dep <= 10)).sum()} ({((valid_dep > 5) & (valid_dep <= 10)).mean()*100:.2f}%)")
    print(f"10 to 15 min count: {((valid_dep > 10) & (valid_dep <= 15)).sum()} ({((valid_dep > 10) & (valid_dep <= 15)).mean()*100:.2f}%)")
    print(f"> 15 min count: {(valid_dep > 15).sum()} ({(valid_dep > 15).mean()*100:.2f}%)")
    print(f"> 20 min count: {(valid_dep > 20).sum()} ({(valid_dep > 20).mean()*100:.2f}%)")
    print(f"P50: {valid_dep.quantile(0.50):.4f}")
    print(f"P90: {valid_dep.quantile(0.90):.4f}")
    print(f"P95: {valid_dep.quantile(0.95):.4f}")
    
    print("\n=== CLIPPED DEPARTURE_DELAY (lower=0.0) ===")
    clipped_dep = valid_dep.clip(lower=0.0)
    print(f"Mean: {clipped_dep.mean():.4f}")
    print(f"Median: {clipped_dep.median():.4f}")
    print(f"Std: {clipped_dep.std():.4f}")
    print(f"P50: {clipped_dep.quantile(0.50):.4f}")
    print(f"P90: {clipped_dep.quantile(0.90):.4f}")
    print(f"P95: {clipped_dep.quantile(0.95):.4f}")
    print(f"Zero count (== 0): {(clipped_dep == 0).sum()} ({(clipped_dep == 0).mean()*100:.2f}%)")
    print(f"<= 5 min: {(clipped_dep <= 5).sum()} ({(clipped_dep <= 5).mean()*100:.2f}%)")
    print(f"> 5 to <= 20 min: {((clipped_dep > 5) & (clipped_dep <= 20)).sum()} ({((clipped_dep > 5) & (clipped_dep <= 20)).mean()*100:.2f}%)")
    print(f"> 20 min: {(clipped_dep > 20).sum()} ({(clipped_dep > 20).mean()*100:.2f}%)")

    # Now inspect multimodal_train.csv
    proc_path = os.path.join(BASE_DIR, "data", "processed", "multimodal_train.csv")
    df_proc = pd.read_csv(proc_path)
    air_proc = df_proc[df_proc['transport_mode'] == 'AIRPLANE']
    print(f"\n=== PROCESSED AIRPLANE in multimodal_train.csv ({len(air_proc)} rows) ===")
    p_delay = air_proc['delay_minutes']
    print(f"Mean: {p_delay.mean():.4f}")
    print(f"Median: {p_delay.median():.4f}")
    print(f"Std: {p_delay.std():.4f}")
    print(f"Min: {p_delay.min():.4f}, Max: {p_delay.max():.4f}")
    print(f"P50: {p_delay.quantile(0.50):.4f}")
    print(f"P90: {p_delay.quantile(0.90):.4f}")
    print(f"P95: {p_delay.quantile(0.95):.4f}")
    print(f"Zero count (== 0): {(p_delay == 0).sum()} ({(p_delay == 0).mean()*100:.2f}%)")
    print(f"<= 5 min: {(p_delay <= 5).sum()} ({(p_delay <= 5).mean()*100:.2f}%)")
    print(f"> 5 to <= 20 min: {((p_delay > 5) & (p_delay <= 20)).sum()} ({((p_delay > 5) & (p_delay <= 20)).mean()*100:.2f}%)")
    print(f"> 20 min: {(p_delay > 20).sum()} ({(p_delay > 20).mean()*100:.2f}%)")
    print(f"Category counts:\n{air_proc['delay_category'].value_counts(normalize=True)*100}")

if __name__ == '__main__':
    audit_aviation_target()
