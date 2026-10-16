import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import joblib

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_processed_data

class TwoPartHurdleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=5.0, clf=None, reg=None):
        self.threshold = threshold
        self.clf = clf if clf is not None else LogisticRegression(max_iter=1000, random_state=42)
        self.reg = reg if reg is not None else HistGradientBoostingRegressor(loss='squared_error', max_iter=100, min_samples_leaf=15, random_state=42)

    def fit(self, X, y):
        is_delayed = (y > self.threshold).astype(int)
        self.clf.fit(X, is_delayed)
        
        delayed_mask = (y > self.threshold)
        if np.sum(delayed_mask) > 10:
            self.reg.fit(X[delayed_mask], y[delayed_mask])
        else:
            self.reg.fit(X, y)
        return self

    def predict(self, X):
        p_delayed = self.clf.predict_proba(X)[:, 1]
        mag_delayed = np.maximum(self.threshold, self.reg.predict(X))
        return p_delayed * mag_delayed

def test_hurdle():
    train_df, val_df, test_df = load_processed_data()
    
    tr_air = train_df[train_df['transport_mode'] == 'AIRPLANE']
    vl_air = val_df[val_df['transport_mode'] == 'AIRPLANE']
    ts_air = test_df[test_df['transport_mode'] == 'AIRPLANE']
    
    prep = joblib.load(os.path.join(BASE_DIR, "models", "preprocessing_pipeline.joblib"))
    
    X_tr = prep.transform(tr_air)
    y_tr = tr_air['delay_minutes'].values
    
    X_vl = prep.transform(vl_air)
    y_vl = vl_air['delay_minutes'].values
    
    X_ts = prep.transform(ts_air)
    y_ts = ts_air['delay_minutes'].values
    
    hurdle = TwoPartHurdleRegressor(threshold=5.0)
    hurdle.fit(X_tr, y_tr)
    
    # Validation evaluation
    pred_vl = hurdle.predict(X_vl)
    mae_vl = mean_absolute_error(y_vl, pred_vl)
    rmse_vl = np.sqrt(mean_squared_error(y_vl, pred_vl))
    medae_vl = median_absolute_error(y_vl, pred_vl)
    
    print("=== Two-Part Hurdle Model Evaluation on Validation Split ===")
    print(f"Validation MAE:  {mae_vl:.4f}")
    print(f"Validation RMSE: {rmse_vl:.4f}")
    print(f"Validation MedAE: {medae_vl:.4f}")
    print(f"Val Pred stats: min={pred_vl.min():.2f}, p10={np.percentile(pred_vl, 10):.2f}, p25={np.percentile(pred_vl, 25):.2f}, p50={np.percentile(pred_vl, 50):.2f}, p75={np.percentile(pred_vl, 75):.2f}, p90={np.percentile(pred_vl, 90):.2f}, max={pred_vl.max():.2f}")
    
    # Test evaluation
    pred_ts = hurdle.predict(X_ts)
    mae_ts = mean_absolute_error(y_ts, pred_ts)
    rmse_ts = np.sqrt(mean_squared_error(y_ts, pred_ts))
    medae_ts = median_absolute_error(y_ts, pred_ts)
    
    print("\n=== Two-Part Hurdle Model Evaluation on Test Split ===")
    print(f"Test MAE:  {mae_ts:.4f}")
    print(f"Test RMSE: {rmse_ts:.4f}")
    print(f"Test MedAE: {medae_ts:.4f}")
    print(f"Test Pred stats: min={pred_ts.min():.2f}, p10={np.percentile(pred_ts, 10):.2f}, p25={np.percentile(pred_ts, 25):.2f}, p50={np.percentile(pred_ts, 50):.2f}, p75={np.percentile(pred_ts, 75):.2f}, p90={np.percentile(pred_ts, 90):.2f}, max={pred_ts.max():.2f}")

    # Compare with Test Baselines
    tr_mean = np.mean(y_tr)
    tr_med = np.median(y_tr)
    mae_mean = mean_absolute_error(y_ts, np.full_like(y_ts, tr_mean))
    rmse_mean = np.sqrt(mean_squared_error(y_ts, np.full_like(y_ts, tr_mean)))
    mae_zero = mean_absolute_error(y_ts, np.zeros_like(y_ts))
    rmse_zero = np.sqrt(mean_squared_error(y_ts, np.zeros_like(y_ts)))
    
    print("\n--- Test Set Baselines Comparison ---")
    print(f"Hurdle Model:    MAE = {mae_ts:.4f}, RMSE = {rmse_ts:.4f}")
    print(f"Mean Baseline:   MAE = {mae_mean:.4f}, RMSE = {rmse_mean:.4f}")
    print(f"Zero Baseline:   MAE = {mae_zero:.4f}, RMSE = {rmse_zero:.4f}")
    print(f"Beats Mean MAE?  {mae_ts < mae_mean} (by {mae_mean - mae_ts:.4f} min)")
    print(f"Beats Zero RMSE? {rmse_ts < rmse_zero} (by {rmse_zero - rmse_ts:.4f} min)")

if __name__ == '__main__':
    test_hurdle()
