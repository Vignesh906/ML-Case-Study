"""
Model Evaluation and Visualization Module
------------------------------------------
Generates visual evaluation artifacts:
- Model comparison bar chart (Macro F1, Weighted F1, Accuracy)
- Confusion matrix heatmaps for each model and best model
- Feature importance bar chart
- Class distribution across transportation modes
- Delay rates by hour and operational factors
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

CLASS_NAMES = ["ON TIME", "MODERATELY DELAYED", "SEVERELY DELAYED"]

def generate_evaluation_plots():
    """Generates all evaluation charts and persists them to results/plots/."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "model_comparison.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Run train_models.py first.")

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    models_info = metrics_data["models"]
    best_model_name = metrics_data["best_model"]
    feature_importances = metrics_data.get("feature_importances", {})

    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Model Comparison Chart
    plt.figure(figsize=(11, 6))
    df_compare = []
    for m_name, m_val in models_info.items():
        df_compare.append({"Model": m_name, "Metric": "Macro F1", "Score": m_val["f1_macro"]})
        df_compare.append({"Model": m_name, "Metric": "Weighted F1", "Score": m_val["f1_weighted"]})
        df_compare.append({"Model": m_name, "Metric": "Accuracy", "Score": m_val["accuracy"]})
    df_compare = pd.DataFrame(df_compare)

    ax = sns.barplot(data=df_compare, x="Model", y="Score", hue="Metric", palette="Blues_r")
    plt.title("Performance Comparison Across All 5 ML Algorithms", fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Model Architecture", fontsize=12, labelpad=10)
    plt.ylabel("Metric Score", fontsize=12, labelpad=10)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title="Metric", loc="upper right")
    plt.tight_layout()
    compare_chart_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(compare_chart_path, dpi=300)
    plt.close()
    print(f"Generated: {compare_chart_path}")

    # 2. Confusion Matrices for all models & Best Model
    for m_name, m_val in models_info.items():
        cm = np.array(m_val["confusion_matrix"])
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        is_best_label = " [BEST MODEL]" if m_name == best_model_name else ""
        plt.title(f"Confusion Matrix: {m_name}{is_best_label}", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Predicted Severity Class", fontsize=11, labelpad=8)
        plt.ylabel("Actual Severity Class", fontsize=11, labelpad=8)
        plt.tight_layout()
        safe_name = m_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{safe_name}.png")
        plt.savefig(cm_path, dpi=300)
        if m_name == best_model_name:
            plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_best.png"), dpi=300)
        plt.close()

    # 3. Feature Importance Bar Chart
    if feature_importances:
        top_items = list(feature_importances.items())[:15]
        feat_df = pd.DataFrame(top_items, columns=["Feature", "Importance"])
        plt.figure(figsize=(10, 7))
        sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
        plt.title(f"Top 15 Most Influential Delay Predictors ({best_model_name})", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Relative Feature Importance Score", fontsize=12, labelpad=10)
        plt.ylabel("Predictor", fontsize=12, labelpad=10)
        plt.tight_layout()
        feat_chart_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.savefig(feat_chart_path, dpi=300)
        plt.close()
        print(f"Generated: {feat_chart_path}")

    # 4. Multimodal Dataset Analytics Charts
    train_csv = os.path.join(PROCESSED_DIR, "multimodal_train.csv")
    if os.path.exists(train_csv):
        df_tr = pd.read_csv(train_csv)

        # 4a. Delay distribution by transport mode
        plt.figure(figsize=(9, 5.5))
        mode_cat = pd.crosstab(df_tr['transport_mode'], df_tr['delay_category'], normalize='index') * 100
        mode_cat.columns = CLASS_NAMES
        mode_cat.plot(kind='bar', stacked=True, color=['#2ecc71', '#f39c12', '#e74c3c'], figsize=(9, 5.5))
        plt.title("Delay Severity Distribution by Transportation Mode", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Transportation Mode", fontsize=11, labelpad=8)
        plt.ylabel("Percentage (%)", fontsize=11, labelpad=8)
        plt.xticks(rotation=0)
        plt.legend(title="Delay Severity", loc="upper right")
        plt.tight_layout()
        dist_path = os.path.join(PLOTS_DIR, "delay_distribution_modes.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()
        print(f"Generated: {dist_path}")

        # 4b. Delay Rate by Hour
        plt.figure(figsize=(10, 5))
        hourly_delays = df_tr.groupby('hour')['delay_category'].apply(lambda x: (x > 0).mean() * 100)
        plt.plot(hourly_delays.index, hourly_delays.values, marker='o', linewidth=2.5, color='#2980b9')
        plt.fill_between(hourly_delays.index, hourly_delays.values, color='#3498db', alpha=0.2)
        plt.title("Delay Frequency Rate Across Operating Hours of Day", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Hour of Day (0–23)", fontsize=11, labelpad=8)
        plt.ylabel("Delayed Trips / Flights (%)", fontsize=11, labelpad=8)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        hourly_path = os.path.join(PLOTS_DIR, "delay_by_hour.png")
        plt.savefig(hourly_path, dpi=300)
        plt.close()
        print(f"Generated: {hourly_path}")

    print("All evaluation plots generated successfully.")

if __name__ == "__main__":
    generate_evaluation_plots()
