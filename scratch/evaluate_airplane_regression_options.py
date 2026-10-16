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

def evaluate_airplane_options():
    train_df, val_df, test_df = load_processed_data()
    
    # Filter for Airplane mode
    tr_air = train_df[train_df['transport_mode'] == 'AIRPLANE'].copy()
    vl_air = val_df[val_df['transport_mode'] == 'AIRPLANE'].copy()
    ts_air = test_df[test_df['transport_mode'] == 'AIRPLANE'].copy()
    
    print(f"Airplane data: Train={len(tr_air)}, Val={len(vl_air)}, Test={len(ts_air)}")
    
    prep_path = os.path.join(BASE_DIR, "models", "preprocessing_pipeline.joblib")
    preprocessor = joblib.load(prep_path)
    
    X_tr = preprocessor.transform(tr_air)
    y_tr = tr_air['delay_minutes'].values
    
    X_vl = preprocessor.transform(vl_air)
    y_vl = vl_air['delay_minutes'].values
    
    X_ts = preprocessor.transform(ts_air)
    y_ts = ts_air['delay_minutes'].values

    print("\n--- 1. Baselines on Validation Set ---")
    tr_med = float(np.median(y_tr))
    tr_mean = float(np.mean(y_tr))
    
    pred_zero_vl = np.zeros_like(y_vl)
    pred_med_vl = np.full_like(y_vl, tr_med)
    pred_mean_vl = np.full_like(y_vl, tr_mean)
    
    print(f"Zero Baseline:    MAE = {mean_absolute_error(y_vl, pred_zero_vl):.4f}, RMSE = {np.sqrt(mean_squared_error(y_vl, pred_zero_vl)):.4f}, MedAE = {median_absolute_error(y_vl, pred_zero_vl):.4f}")
    print(f"Median Baseline:  MAE = {mean_absolute_error(y_vl, pred_med_vl):.4f}, RMSE = {np.sqrt(mean_squared_error(y_vl, pred_med_vl)):.4f}, MedAE = {median_absolute_error(y_vl, pred_med_vl):.4f}")
    print(f"Mean Baseline:    MAE = {mean_absolute_error(y_vl, pred_mean_vl):.4f}, RMSE = {np.sqrt(mean_squared_error(y_vl, pred_mean_vl)):.4f}, MedAE = {median_absolute_error(y_vl, pred_mean_vl):.4f}")

    print("\n--- 2. Option A: HistGradientBoostingRegressor (squared_error - current) ---")
    reg_sq = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, min_samples_leaf=20, random_state=42)
    reg_sq.fit(X_tr, y_tr)
    pred_sq_vl = np.maximum(0, reg_sq.predict(X_vl))
    print(f"Squared Error:   MAE = {mean_absolute_error(y_vl, pred_sq_vl):.4f}, RMSE = {np.sqrt(mean_squared_error(y_vl, pred_sq_vl)):.4f}, MedAE = {median_absolute_error(y_vl, pred_sq_vl):.4f}")
    print(f"  Prediction stats: min={pred_sq_vl.min():.2f}, p10={np.percentile(pred_sq_vl, 10):.2f}, p50={np.percentile(pred_sq_vl, 50):.2f}, mean={pred_sq_vl.mean():.2f}, p90={np.percentile(pred_sq_vl, 90):.2f}, max={pred_sq_vl.max():.2f}")

    print("\n--- 3. Option B: HistGradientBoostingRegressor (absolute_error - L1 / Median) ---")
    reg_abs = HistGradientBoostingRegressor(loss='absolute_error', max_iter=100, min_samples_leaf=20, random_state=42)
    reg_abs.fit(X_tr, y_tr)
    pred_abs_vl = np.maximum(0, reg_abs.predict(X_vl))
    print(f"Absolute Error:  MAE = {mean_absolute_error(y_vl, pred_abs_vl):.4f}, RMSE = {np.sqrt(mean_squared_error(y_vl, pred_abs_vl)):.4f}, MedAE = {median_absolute_error(y_vl, pred_abs_vl):.4f}")
    print(f"  Prediction stats: min={pred_abs_vl.min():.2f}, p10={np.percentile(pred_abs_vl, 10):.2f}, p50={np.percentile(pred_abs_vl, 50):.2f}, mean={pred_abs_vl.mean():.2f}, p90={np.percentile(pred_abs_vl, 90):.2f}, max={pred_abs_vl.max():.2f}")

    print("\n--- 4. Option C: Two-Part Hurdle Model (P(delay > 5) * E[delay | delay > 5]) ---")
    # Part 1: Prob of positive delay (> 5 min)
    is_pos_tr = (y_tr > 5.0).astype(int)
    from sklearn.linear_model import LogisticRegression
    clf_hurdle = LogisticRegression(max_iter=1000, random_state=42)
    clf_hurdle.fit(X_tr, is_pos_tr)
    p_pos_vl = clf_hurdle.predict_proba(X_vl)[:, 1]
    
    # Part 2: Magnitude for positive cases
    pos_mask_tr = (y_tr > 5.0)
    reg_pos = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, min_samples_leaf=15, random_state=42)
    reg_pos.fit(X_tr[pos_mask_tr], y_tr[pos_mask_tr])
    mag_pos_vl = np.maximum(5.0, reg_pos.predict(X_vl))
    
    # Expected delay = p_pos * mag_pos (or 0 if low prob)
    pred_hurdle_vl = p_pos_vl * mag_pos_vl
    print(f"Hurdle Model:    MAE = {mean_absolute_error(y_vl, pred_hurdle_vl):.4f}, RMSE = {np.sqrt(mean_squared_error(y_vl, pred_hurdle_vl)):.4f}, MedAE = {median_absolute_error(y_vl, pred_hurdle_vl):.4f}")
    print(f"  Prediction stats: min={pred_hurdle_vl.min():.2f}, p10={np.percentile(pred_hurdle_vl, 10):.2f}, p50={np.percentile(pred_hurdle_vl, 50):.2f}, mean={pred_hurdle_vl.mean():.2f}, p90={np.percentile(pred_hurdle_vl, 90):.2f}, max={pred_hurdle_vl.max():.2f}")

    print("\n--- 5. Option D: Log-Transformed Regression (log1p target) ---")
    y_tr_log = np.log1p(y_tr)
    reg_log = HistGradientBoostingRegressor(loss='squared_error', max_iter=100, min_samples_leaf=20, random_state=42)
    reg_log.fit(X_tr, y_tr_log)
    pred_log_vl = np.expm1(np.maximum(0, reg_log.predict(X_vl)))
    print(f"Log-Transformed: MAE = {mean_absolute_error(y_vl, pred_log_vl):.4f}, RMSE = {np.sqrt(mean_squared_error(y_vl, pred_log_vl)):.4f}, MedAE = {median_absolute_error(y_vl, pred_log_vl):.4f}")
    print(f"  Prediction stats: min={pred_log_vl.min():.2f}, p10={np.percentile(pred_log_vl, 10):.2f}, p50={np.percentile(pred_log_vl, 50):.2f}, mean={pred_log_vl.mean():.2f}, p90={np.percentile(pred_log_vl, 90):.2f}, max={pred_log_vl.max():.2f}")

if __name__ == '__main__':
    evaluate_airplane_options()
