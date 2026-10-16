import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_processed_data

def test_models():
    train_df, val_df, test_df = load_processed_data()
    
    tr_air = train_df[train_df['transport_mode'] == 'AIRPLANE'].copy()
    vl_air = val_df[val_df['transport_mode'] == 'AIRPLANE'].copy()
    ts_air = test_df[test_df['transport_mode'] == 'AIRPLANE'].copy()
    
    prep = joblib.load(os.path.join(BASE_DIR, "models", "preprocessing_pipeline.joblib"))
    
    X_tr = prep.transform(tr_air)
    y_tr = tr_air['delay_minutes'].values
    
    X_vl = prep.transform(vl_air)
    y_vl = vl_air['delay_minutes'].values
    
    X_ts = prep.transform(ts_air)
    y_ts = ts_air['delay_minutes'].values

    # 1. Poisson Loss (Standard for count/zero-inflated non-negative delay distributions)
    print("\n--- Model: HistGradientBoostingRegressor(loss='poisson') ---")
    reg_pois = HistGradientBoostingRegressor(loss='poisson', max_iter=100, min_samples_leaf=20, random_state=42)
    reg_pois.fit(X_tr, y_tr)
    pred_pois_vl = reg_pois.predict(X_vl)
    print(f"Poisson Val MAE:  {mean_absolute_error(y_vl, pred_pois_vl):.4f}")
    print(f"Poisson Val RMSE: {np.sqrt(mean_squared_error(y_vl, pred_pois_vl)):.4f}")
    print(f"Poisson Val MedAE: {median_absolute_error(y_vl, pred_pois_vl):.4f}")
    print(f"Poisson Pred stats: min={pred_pois_vl.min():.2f}, p10={np.percentile(pred_pois_vl, 10):.2f}, p25={np.percentile(pred_pois_vl, 25):.2f}, p50={np.percentile(pred_pois_vl, 50):.2f}, p75={np.percentile(pred_pois_vl, 75):.2f}, p90={np.percentile(pred_pois_vl, 90):.2f}, max={pred_pois_vl.max():.2f}")

    # 2. Two-Part Hurdle Model (Calibrated P(delayed) * E[delay | delayed])
    print("\n--- Model: Two-Part Hurdle Model ---")
    # Part 1: Probability of delay > 5 min
    from sklearn.linear_model import LogisticRegression
    y_tr_hurdle = (y_tr > 5.0).astype(int)
    clf_hurdle = LogisticRegression(max_iter=1000, random_state=42)
    clf_hurdle.fit(X_tr, y_tr_hurdle)
    p_delayed_vl = clf_hurdle.predict_proba(X_vl)[:, 1]
    
    # Part 2: Magnitude given delay > 5 min
    pos_mask_tr = (y_tr > 5.0)
    reg_cond = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, min_samples_leaf=15, random_state=42)
    reg_cond.fit(X_tr[pos_mask_tr], y_tr[pos_mask_tr])
    cond_mag_vl = np.maximum(5.0, reg_cond.predict(X_vl))
    
    pred_hurdle_vl = p_delayed_vl * cond_mag_vl
    print(f"Hurdle Val MAE:  {mean_absolute_error(y_vl, pred_hurdle_vl):.4f}")
    print(f"Hurdle Val RMSE: {np.sqrt(mean_squared_error(y_vl, pred_hurdle_vl)):.4f}")
    print(f"Hurdle Val MedAE: {median_absolute_error(y_vl, pred_hurdle_vl):.4f}")
    print(f"Hurdle Pred stats: min={pred_hurdle_vl.min():.2f}, p10={np.percentile(pred_hurdle_vl, 10):.2f}, p25={np.percentile(pred_hurdle_vl, 25):.2f}, p50={np.percentile(pred_hurdle_vl, 50):.2f}, p75={np.percentile(pred_hurdle_vl, 75):.2f}, p90={np.percentile(pred_hurdle_vl, 90):.2f}, max={pred_hurdle_vl.max():.2f}")

    # 3. Quantile Median (quantile=0.50) + Quantile High-End (quantile=0.90)
    print("\n--- Model: Quantile Regressors ---")
    q50 = HistGradientBoostingRegressor(loss='quantile', quantile=0.50, max_iter=100, min_samples_leaf=20, random_state=42)
    q50.fit(X_tr, y_tr)
    pred_q50_vl = np.maximum(0, q50.predict(X_vl))
    print(f"Q50 Pred stats: min={pred_q50_vl.min():.2f}, p10={np.percentile(pred_q50_vl, 10):.2f}, p50={np.percentile(pred_q50_vl, 50):.2f}, p90={np.percentile(pred_q50_vl, 90):.2f}, max={pred_q50_vl.max():.2f}")

    q90 = HistGradientBoostingRegressor(loss='quantile', quantile=0.90, max_iter=100, min_samples_leaf=20, random_state=42)
    q90.fit(X_tr, y_tr)
    pred_q90_vl = np.maximum(0, q90.predict(X_vl))
    print(f"Q90 Pred stats: min={pred_q90_vl.min():.2f}, p10={np.percentile(pred_q90_vl, 10):.2f}, p50={np.percentile(pred_q90_vl, 50):.2f}, p90={np.percentile(pred_q90_vl, 90):.2f}, max={pred_q90_vl.max():.2f}")

if __name__ == '__main__':
    test_models()
