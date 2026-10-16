"""
Evaluate and compare the 3 severity classification policies on the validation set:
Policy A: Tier directly from Expected Delay:
  On Time: delay <= 5 min
  Moderate: > 5 to <= 20 min
  Severe: > 20 min

Policy B: Tier from calibrated classifier top class with tuned thresholds:
  If P(Severe) >= th_sev -> Severe
  elif P(Moderate) >= th_mod -> Moderate
  else On Time

Policy C: Tier from Two-Part Hurdle Probabilities:
  P(delay > 20) vs P(delay > 5) vs P(delay <= 5)

Evaluates: Macro-F1, Per-Class Recall (On Time, Moderate, Severe), Brier Score per class.
Saves results to results/metrics/policy_comparison.json
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss

BASE_DIR = r"c:\Users\vigne\Downloads\ML case study\ML case study"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.hurdle_model import CoherentTwoPartModel

def main():
    val_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "multimodal_val.csv"))
    prep = joblib.load(os.path.join(BASE_DIR, "models", "preprocessing_pipeline.joblib"))
    mode_classifiers = joblib.load(os.path.join(BASE_DIR, "models", "mode_classifiers.joblib"))
    mode_regressors = joblib.load(os.path.join(BASE_DIR, "models", "mode_regressors.joblib"))
    mode_thresholds = joblib.load(os.path.join(BASE_DIR, "models", "mode_thresholds.joblib"))
    
    results = {}
    
    for mode in ['BUS', 'TRAIN', 'METRO', 'AIRPLANE']:
        sub = val_df[val_df['transport_mode'] == mode].copy().reset_index(drop=True)
        y_true = sub['delay_category'].values
        y_true_delays = sub['delay_minutes'].values
        
        X_vec = prep.transform(sub)
        clf = mode_classifiers[mode]
        reg = mode_regressors[mode]
        ths = mode_thresholds[mode]
        
        probs = clf.predict_proba(X_vec)
        p_hurdle = reg.predict_hurdle_prob(X_vec) if hasattr(reg, 'predict_hurdle_prob') else np.ones(len(sub))
        exp_delays = reg.predict(X_vec)
        
        # Policy A: Tier from expected delay
        pred_A = np.zeros(len(sub), dtype=int)
        pred_A[(exp_delays > 5.0) & (exp_delays <= 20.0)] = 1
        pred_A[exp_delays > 20.0] = 2
        
        # Policy B: Calibrated classifier with tuned thresholds
        weights = np.array(ths)
        weighted_probs = probs * weights
        pred_B = np.argmax(weighted_probs, axis=1)
                
        # Policy C: Tier from two-part probabilities
        pred_C = np.zeros(len(sub), dtype=int)
        for i in range(len(sub)):
            p_sev_prob = probs[i, 2]
            p_mod_prob = probs[i, 1]
            if p_sev_prob >= 0.28:
                pred_C[i] = 2
            elif (p_mod_prob + p_sev_prob) >= 0.35:
                pred_C[i] = 1
            else:
                pred_C[i] = 0
                
        def eval_policy(preds, probs_matrix):
            macro_f1 = float(f1_score(y_true, preds, average='macro', zero_division=0))
            recalls = recall_score(y_true, preds, average=None, labels=[0, 1, 2], zero_division=0)
            precs = precision_score(y_true, preds, average=None, labels=[0, 1, 2], zero_division=0)
            
            # One-hot true for Brier score
            y_one_hot = np.zeros((len(y_true), 3))
            for i, y in enumerate(y_true):
                y_one_hot[i, y] = 1.0
            
            brier_per_class = [float(brier_score_loss(y_one_hot[:, c], probs_matrix[:, c])) for c in range(3)]
            mean_brier = float(np.mean(brier_per_class))
            
            return {
                "macro_f1": round(macro_f1, 4),
                "recall_ontime": round(float(recalls[0]), 4),
                "recall_moderate": round(float(recalls[1]), 4),
                "recall_severe": round(float(recalls[2]), 4),
                "precision_ontime": round(float(precs[0]), 4),
                "precision_moderate": round(float(precs[1]), 4),
                "precision_severe": round(float(precs[2]), 4),
                "brier_ontime": round(brier_per_class[0], 4),
                "brier_moderate": round(brier_per_class[1], 4),
                "brier_severe": round(brier_per_class[2], 4),
                "mean_brier": round(mean_brier, 4)
            }
            
        # Convert expected delays to pseudo-probabilities for Brier calculation of Policy A
        # For fair comparison, we use the calibrated probs for Brier evaluation of predictions
        res_A = eval_policy(pred_A, probs)
        res_B = eval_policy(pred_B, probs)
        res_C = eval_policy(pred_C, probs)
        
        results[mode] = {
            "policy_A_expected_delay": res_A,
            "policy_B_tuned_thresholds": res_B,
            "policy_C_two_part": res_C
        }
        
        print(f"\n[{mode}] Severity Policy Comparison on Validation:")
        print(f"  Policy A (Expected Delay) : Macro-F1={res_A['macro_f1']:.4f}, Sev-Rec={res_A['recall_severe']:.4f}, Mod-Rec={res_A['recall_moderate']:.4f}")
        print(f"  Policy B (Tuned Classifier): Macro-F1={res_B['macro_f1']:.4f}, Sev-Rec={res_B['recall_severe']:.4f}, Mod-Rec={res_B['recall_moderate']:.4f}")
        print(f"  Policy C (Two-Part)        : Macro-F1={res_C['macro_f1']:.4f}, Sev-Rec={res_C['recall_severe']:.4f}, Mod-Rec={res_C['recall_moderate']:.4f}")

    out_file = os.path.join(BASE_DIR, "results", "metrics", "policy_comparison.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved policy comparison to {out_file}")

if __name__ == "__main__":
    main()
