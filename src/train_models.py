"""
Model Training, Calibration, Benchmarking, and Evaluation Module
----------------------------------------------------------------
Trains and rigorously benchmarks 5 machine learning classification models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (Linear SVM, calibrated)
5. K-Nearest Neighbors (KNN)

Plus mode-specific continuous delay and quantile regression models:
6. Mode-Specific HistGradientBoostingRegressor (Expected Delay, Typical Median P50, High-End Risk P90)
   per transportation mode (Bus, Train, Metro, Airplane)

Performs comprehensive multiclass evaluation (Accuracy, Precision, Recall,
Macro F1, Weighted F1, Confusion Matrix) both overall and stratified by
transportation mode (Bus, Train, Metro, Airplane), benchmarks against
majority-class and median/mean regression baselines, and calibrates probabilities on validation data.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, brier_score_loss
)
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_processed_data
from src.preprocessing import MultimodalPreprocessor, save_preprocessor
from src.hurdle_model import TwoPartHurdleRegressor

MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
REPORTS_DIR = os.path.join(BASE_DIR, "results", "reports")

CLASS_NAMES = ["ON TIME", "MODERATELY DELAYED", "SEVERELY DELAYED"]
MODES = ["BUS", "TRAIN", "METRO", "AIRPLANE"]

def get_raw_model_candidates():
    """Initializes the 5 required machine learning algorithms."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(
                dual=False,
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            )
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            n_jobs=-1
        )
    }
    return models

def compute_metrics_for_subset(y_true, y_pred, y_probs=None):
    """Computes all classification metrics and calibration Brier score on a subset."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    cls_rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # Per-class metrics
    per_class = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        per_class[c_name] = {
            "precision": round(float(cls_rep[c_name]['precision']), 4),
            "recall": round(float(cls_rep[c_name]['recall']), 4),
            "f1_score": round(float(cls_rep[c_name]['f1-score']), 4),
            "support": int(cls_rep[c_name]['support'])
        }

    brier = None
    if y_probs is not None:
        brier = {}
        for c_idx, c_name in enumerate(CLASS_NAMES):
            y_binary = (y_true == c_idx).astype(int)
            brier[c_name] = round(float(brier_score_loss(y_binary, y_probs[:, c_idx])), 4)

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_macro": round(float(rec_macro), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_macro": round(float(f1_m), 4),
        "f1_weighted": round(float(f1_w), 4),
        "confusion_matrix": cm,
        "classification_report": cls_rep,
        "per_class": per_class,
        "brier_scores": brier
    }

def train_and_evaluate_all():
    """
    Main training workflow executing preprocessing, training, calibration on validation data,
    evaluation on held-out test data, mode-specific breakdown, continuous delay regression,
    and artifact persistence.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 65)
    print("STARTING MULTIMODAL MODEL TRAINING & EVALUATION WORKFLOW")
    print("=" * 65)

    # 1. Load data
    train_df, val_df, test_df = load_processed_data()
    print(f"Data Loaded: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # Report class distributions per split
    splits_summary = {}
    for s_name, df_s in [('train', train_df), ('val', val_df), ('test', test_df)]:
        splits_summary[s_name] = {}
        for m in MODES:
            sub = df_s[df_s['transport_mode'] == m]
            splits_summary[s_name][m] = {
                "count": len(sub),
                "mean_delay": round(float(sub['delay_minutes'].mean()), 2),
                "median_delay": round(float(sub['delay_minutes'].median()), 2),
                "pct_zero": round(float((sub['delay_minutes'] == 0).mean() * 100), 2),
                "class_proportions": {
                    CLASS_NAMES[k]: round(float(v), 4)
                    for k, v in sub['delay_category'].value_counts(normalize=True).items()
                }
            }

    # 2. Fit Preprocessor
    print("\nFitting MultimodalPreprocessor pipeline with mode-interaction features...")
    preprocessor = MultimodalPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    y_train = train_df['delay_category'].values
    y_val = val_df['delay_category'].values
    y_test = test_df['delay_category'].values

    y_train_reg = train_df['delay_minutes'].values
    y_val_reg = val_df['delay_minutes'].values
    y_test_reg = test_df['delay_minutes'].values

    feature_names = preprocessor.get_feature_names()
    print(f"Feature transformation complete. Input dimensions: {X_train.shape[1]} features.")

    # Save Preprocessor
    prep_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    save_preprocessor(preprocessor, prep_path)

    # 3. Baselines on Held-out Test Set
    majority_class = 0 # ON TIME
    baseline_pred = np.full_like(y_test, majority_class)
    overall_baseline_metrics = {
        "accuracy": round(float(accuracy_score(y_test, baseline_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, baseline_pred, average='macro', zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_test, baseline_pred, average='macro', zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, baseline_pred, average='macro', zero_division=0)), 4)
    }

    mode_baselines = {}
    for m in MODES:
        sub_mask = (test_df['transport_mode'] == m).values
        sub_y = y_test[sub_mask]
        sub_pred = np.full_like(sub_y, majority_class)
        mode_baselines[m] = {
            "sample_count": int(np.sum(sub_mask)),
            "accuracy": round(float(accuracy_score(sub_y, sub_pred)), 4),
            "f1_macro": round(float(f1_score(sub_y, sub_pred, average='macro', zero_division=0)), 4),
            "precision_macro": round(float(precision_score(sub_y, sub_pred, average='macro', zero_division=0)), 4),
            "recall_macro": round(float(recall_score(sub_y, sub_pred, average='macro', zero_division=0)), 4)
        }

    print("\n--- Majority-Class Naive Baselines (Predicting All 'ON TIME') ---")
    print(f"  Overall Test Accuracy: {overall_baseline_metrics['accuracy']:.4f}, Macro-F1: {overall_baseline_metrics['f1_macro']:.4f}")
    for m in MODES:
        print(f"  {m:<9} Test Accuracy: {mode_baselines[m]['accuracy']:.4f}, Macro-F1: {mode_baselines[m]['f1_macro']:.4f}")

    # 4. Train and Calibrate all 5 classification models
    raw_models = get_raw_model_candidates()
    results = {}
    trained_models = {}
    reports = {}
    mode_evaluations = {}
    reliability_curves = {}

    print("\nTraining, calibrating on validation split, and evaluating 5 classification algorithms...")
    for name, raw_model in raw_models.items():
        print(f"\n--- Training & Calibrating: {name} ---")
        t0 = time.time()
        raw_model.fit(X_train, y_train)

        # Calibrate probabilities using validation set (Platt scaling / Sigmoid)
        if name != "Linear SVM (calibrated)":
            calibrated_model = CalibratedClassifierCV(estimator=raw_model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_val, y_val)
        else:
            calibrated_model = raw_model

        train_time = round(time.time() - t0, 3)

        # Test set inference
        y_pred = calibrated_model.predict(X_test)
        y_probs = calibrated_model.predict_proba(X_test) if hasattr(calibrated_model, "predict_proba") else None

        global_metrics = compute_metrics_for_subset(y_test, y_pred, y_probs)
        global_metrics["model_name"] = name
        global_metrics["train_time_sec"] = train_time

        # Compute calibration curve for On-Time class
        if y_probs is not None:
            prob_true, prob_pred = calibration_curve((y_test == 0).astype(int), y_probs[:, 0], n_bins=10)
            reliability_curves[name] = {
                "prob_true": [round(float(v), 4) for v in prob_true],
                "prob_pred": [round(float(v), 4) for v in prob_pred]
            }

        results[name] = {
            "model_name": name,
            "train_time_sec": train_time,
            "accuracy": global_metrics["accuracy"],
            "precision_macro": global_metrics["precision_macro"],
            "precision_weighted": global_metrics["precision_weighted"],
            "recall_macro": global_metrics["recall_macro"],
            "recall_weighted": global_metrics["recall_weighted"],
            "f1_macro": global_metrics["f1_macro"],
            "f1_weighted": global_metrics["f1_weighted"],
            "confusion_matrix": global_metrics["confusion_matrix"],
            "per_class": global_metrics["per_class"],
            "brier_scores": global_metrics["brier_scores"]
        }
        trained_models[name] = calibrated_model
        reports[name] = global_metrics["classification_report"]

        # Stratified evaluation by transportation mode
        mode_evaluations[name] = {}
        for m in MODES:
            idx_m = (test_df['transport_mode'] == m).values
            m_probs = y_probs[idx_m] if y_probs is not None else None
            m_metrics = compute_metrics_for_subset(y_test[idx_m], y_pred[idx_m], m_probs)
            mode_evaluations[name][m] = {
                "sample_count": int(np.sum(idx_m)),
                "accuracy": m_metrics["accuracy"],
                "precision_macro": m_metrics["precision_macro"],
                "recall_macro": m_metrics["recall_macro"],
                "f1_macro": m_metrics["f1_macro"],
                "f1_weighted": m_metrics["f1_weighted"],
                "confusion_matrix": m_metrics["confusion_matrix"],
                "per_class": m_metrics["per_class"],
                "brier_scores": m_metrics["brier_scores"]
            }

        print(f"Results for {name}:")
        print(f"  Overall Accuracy:  {global_metrics['accuracy']:.4f}")
        print(f"  Overall Macro F1:  {global_metrics['f1_macro']:.4f}  [Primary Selection Metric]")
        print(f"  Training Time:     {train_time:.2f}s")
        if global_metrics["brier_scores"]:
            print(f"  Brier Scores:      {global_metrics['brier_scores']}")
        print("  Mode Macro F1: " + ", ".join([f"{m}: {mode_evaluations[name][m]['f1_macro']:.4f}" for m in MODES]))

    # 5. Train Mode-Specific Continuous Delay Regressors and Quantile Models
    print("\n--- Training Mode-Specific Continuous Delay & Quantile Regressors ---")
    t0_reg = time.time()
    mode_regressors = {}
    mode_quantiles_p50 = {}
    mode_quantiles_p90 = {}
    reg_mode_metrics = {}
    all_test_preds = np.zeros(len(test_df))

    for m in MODES:
        idx_tr = (train_df['transport_mode'] == m).values
        idx_ts = (test_df['transport_mode'] == m).values

        # 1. Expected Value (Two-part hurdle for zero-inflated Airplane, Gradient Boosting for Transit)
        if m == 'AIRPLANE':
            m_reg = TwoPartHurdleRegressor(
                threshold=5.0,
                min_samples_leaf=15,
                max_iter=150,
                random_state=42
            )
        else:
            m_reg = HistGradientBoostingRegressor(
                loss='squared_error',
                max_iter=150,
                min_samples_leaf=15,
                learning_rate=0.08,
                random_state=42
            )
        m_reg.fit(X_train[idx_tr], y_train_reg[idx_tr])
        mode_regressors[m] = m_reg

        # 2. Typical Median (Quantile q=0.5)
        m_q50 = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=0.5,
            max_iter=100,
            min_samples_leaf=15,
            learning_rate=0.08,
            random_state=42
        )
        m_q50.fit(X_train[idx_tr], y_train_reg[idx_tr])
        mode_quantiles_p50[m] = m_q50

        # 3. High-End Risk (Quantile q=0.9)
        m_q90 = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=0.9,
            max_iter=100,
            min_samples_leaf=15,
            learning_rate=0.08,
            random_state=42
        )
        m_q90.fit(X_train[idx_tr], y_train_reg[idx_tr])
        mode_quantiles_p90[m] = m_q90

        # Evaluate on Test Set
        m_pred = np.maximum(m_reg.predict(X_test[idx_ts]), 0.0)
        m_p50 = np.maximum(m_q50.predict(X_test[idx_ts]), 0.0)
        m_p90 = np.maximum(m_q90.predict(X_test[idx_ts]), 0.0)

        all_test_preds[idx_ts] = m_pred
        y_true_m = y_test_reg[idx_ts]

        m_mae = round(float(mean_absolute_error(y_true_m, m_pred)), 2)
        m_rmse = round(float(np.sqrt(mean_squared_error(y_true_m, m_pred))), 2)

        # Baseline comparisons on SAME held-out test set
        tr_med_val = float(np.median(y_train_reg[idx_tr]))
        tr_mean_val = float(np.mean(y_train_reg[idx_tr]))
        base_mae_med = round(float(mean_absolute_error(y_true_m, np.full_like(y_true_m, tr_med_val))), 2)
        base_mae_mean = round(float(mean_absolute_error(y_true_m, np.full_like(y_true_m, tr_mean_val))), 2)
        base_rmse_med = round(float(np.sqrt(mean_squared_error(y_true_m, np.full_like(y_true_m, tr_med_val)))), 2)

        reg_mode_metrics[m] = {
            "sample_count": int(np.sum(idx_ts)),
            "mae_minutes": m_mae,
            "rmse_minutes": m_rmse,
            "baseline_median_mae": base_mae_med,
            "baseline_mean_mae": base_mae_mean,
            "baseline_rmse": base_rmse_med,
            "beats_baseline": bool(m_mae < base_mae_med),
            "typical_median_pred_mean": round(float(np.mean(m_p50)), 2),
            "high_end_p90_pred_mean": round(float(np.mean(m_p90)), 2),
            "actual_test_median": round(float(np.median(y_true_m)), 2),
            "actual_test_p90": round(float(np.percentile(y_true_m, 90)), 2)
        }

    reg_train_time = round(time.time() - t0_reg, 3)
    overall_mae = round(float(mean_absolute_error(y_test_reg, all_test_preds)), 2)
    overall_rmse = round(float(np.sqrt(mean_squared_error(y_test_reg, all_test_preds))), 2)

    # Save mode-specific regressors dictionary
    joblib.dump(mode_regressors, os.path.join(MODELS_DIR, "delay_regressors.joblib"))
    joblib.dump(mode_quantiles_p50, os.path.join(MODELS_DIR, "delay_quantiles_p50.joblib"))
    joblib.dump(mode_quantiles_p90, os.path.join(MODELS_DIR, "delay_quantiles_p90.joblib"))
    joblib.dump(mode_regressors["AIRPLANE"], os.path.join(MODELS_DIR, "delay_regressor.joblib"))

    regression_payload = {
        "model_architecture": "Mode-Specific HistGradientBoostingRegressor & Quantile Ensemble",
        "train_time_sec": reg_train_time,
        "overall_mae_minutes": overall_mae,
        "overall_rmse_minutes": overall_rmse,
        "mode_metrics": reg_mode_metrics
    }

    print(f"\nMode-Specific Delay Regressors trained in {reg_train_time:.2f}s:")
    print(f"  Overall Multimodal MAE:  {overall_mae} minutes")
    print(f"  Overall Multimodal RMSE: {overall_rmse} minutes")
    for m in MODES:
        rm = reg_mode_metrics[m]
        status_str = "BEATS BASELINE" if rm['beats_baseline'] else "LOSES TO BASELINE"
        print(f"  {m:<9} Model MAE: {rm['mae_minutes']} min (Baseline Med: {rm['baseline_median_mae']} min) -> {status_str}")

    # 6. Automatic Best Model Selection based on Macro F1
    best_model_name = max(results, key=lambda k: results[k]['f1_macro'])
    best_model = trained_models[best_model_name]
    print("\n" + "=" * 65)
    print(f"BEST CLASSIFIER SELECTED: {best_model_name}")
    print(f"Best Macro F1 Score: {results[best_model_name]['f1_macro']:.4f}")
    print("=" * 65)

    # 7. Extract Feature Importances (Model-Derived & Permutation Importance)
    print("\nComputing Permutation Feature Importance on held-out test data...")
    perm_res = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42)
    sorted_idx = np.argsort(perm_res.importances_mean)[::-1]
    perm_importances = {
        feature_names[i]: round(float(perm_res.importances_mean[i]), 5)
        for i in sorted_idx[:15]
    }

    # Extract base tree/linear feature importances
    base_clf = getattr(best_model, 'estimator', best_model)
    feature_importances = {}
    if hasattr(base_clf, "feature_importances_"):
        raw_imp = base_clf.feature_importances_
        feature_importances = {feat: round(float(imp), 4) for feat, imp in zip(feature_names, raw_imp)}
    elif hasattr(base_clf, "coef_"):
        coef_mean = np.mean(np.abs(base_clf.coef_), axis=0)
        feature_importances = {feat: round(float(imp), 4) for feat, imp in zip(feature_names, coef_mean)}
    else:
        feature_importances = {feat: float(perm_importances.get(feat, 0.01)) for feat in feature_names}

    sorted_importance = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    # Save Best Model Artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(trained_models, os.path.join(MODELS_DIR, "all_models.joblib"))

    # Complete Metrics Payload for Models page & API
    metrics_payload = {
        "best_model": best_model_name,
        "best_macro_f1": results[best_model_name]['f1_macro'],
        "feature_names": feature_names,
        "feature_importances": sorted_importance,
        "permutation_importances": perm_importances,
        "class_names": CLASS_NAMES,
        "class_base_rates": {
            CLASS_NAMES[c]: round(float((y_test == c).mean() * 100), 2)
            for c in range(3)
        },
        "overall_baseline": overall_baseline_metrics,
        "mode_baselines": mode_baselines,
        "models": results,
        "mode_evaluations": mode_evaluations,
        "regression_metrics": regression_payload,
        "reliability_curves": reliability_curves,
        "splits_summary": splits_summary
    }

    with open(os.path.join(METRICS_DIR, "model_comparison.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(os.path.join(METRICS_DIR, "classification_reports.json"), "w") as f:
        json.dump(reports, f, indent=2)

    with open(os.path.join(METRICS_DIR, "regression_metrics.json"), "w") as f:
        json.dump(regression_payload, f, indent=2)

    # 8. Write comprehensive textual training summary report
    summary_txt = os.path.join(REPORTS_DIR, "training_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("MULTIMODAL TRANSPORTATION DELAY PREDICTION - COMPREHENSIVE BENCHMARK REPORT\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Total Training Samples:   {len(train_df):,}\n")
        f.write(f"Total Validation Samples: {len(val_df):,}\n")
        f.write(f"Total Test Samples:       {len(test_df):,} (3,000 per mode: Bus, Train, Metro, Airplane)\n\n")

        f.write("1. ALGORITHM CANDIDATE PERFORMANCE COMPARISON ON HELD-OUT TEST DATA:\n")
        f.write(f"{'Algorithm Candidate':<28} {'Accuracy':<10} {'Macro Prec':<12} {'Macro Rec':<11} {'Macro F1':<11} {'Weighted F1':<12}\n")
        f.write("-" * 85 + "\n")
        for m_name, m_res in results.items():
            f.write(f"{m_name:<28} {m_res['accuracy']:<10.4f} {m_res['precision_macro']:<12.4f} {m_res['recall_macro']:<11.4f} {m_res['f1_macro']:<11.4f} {m_res['f1_weighted']:<12.4f}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Naive Majority Baseline':<28} {overall_baseline_metrics['accuracy']:<10.4f} {overall_baseline_metrics['precision_macro']:<12.4f} {overall_baseline_metrics['recall_macro']:<11.4f} {overall_baseline_metrics['f1_macro']:<11.4f} N/A\n\n")

        f.write(f"Best Classification Model: {best_model_name} (Macro F1 = {results[best_model_name]['f1_macro']:.4f})\n\n")

        f.write("2. REGRESSION VS BASELINE BENCHMARK ON HELD-OUT TEST DATA:\n")
        f.write(f"{'Mode':<10} {'Model MAE':<12} {'Median Baseline':<17} {'Mean Baseline':<15} {'Status':<16}\n")
        f.write("-" * 70 + "\n")
        for m in MODES:
            rm = reg_mode_metrics[m]
            st = "BEATS BASELINE" if rm['beats_baseline'] else "LOSES TO BASELINE"
            f.write(f"{m:<10} {rm['mae_minutes']:<12.2f} {rm['baseline_median_mae']:<17.2f} {rm['baseline_mean_mae']:<15.2f} {st:<16}\n")
        f.write("-" * 70 + "\n\n")

    print(f"\nTraining and benchmarking completed successfully! Metrics written to {METRICS_DIR}")
    return metrics_payload

if __name__ == "__main__":
    train_and_evaluate_all()
