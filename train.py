"""
Master Training Pipeline Entry Point
------------------------------------
Executes the complete, end-to-end multimodal training workflow:
1. Inspects and loads datasets
2. Preprocesses and builds multimodal splits
3. Engineers leakage-free features
4. Trains 5 candidate ML models with class imbalance handling
5. Evaluates models using multiclass metrics (Accuracy, Precision, Recall, Macro F1, Weighted F1)
6. Automatically selects the best model based on Macro F1
7. Persists trained model artifacts and preprocessing pipeline
8. Generates visual evaluation plots and performance reports
"""

import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import build_multimodal_datasets
from src.train_models import train_and_evaluate_all
from src.evaluate_models import generate_evaluation_plots

def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("MULTIMODAL TRANSPORTATION DELAY PREDICTION - END-TO-END TRAINING")
    print("=" * 70)

    # Step 1-5: Check/build datasets
    print("\n[STEP 1/3] Preparing and harmonizing multimodal datasets...")
    build_multimodal_datasets(samples_per_mode_train=20000, samples_per_mode_val=3000, samples_per_mode_test=3000)

    # Step 6-10: Train & Evaluate all 5 models, select best model, save artifacts
    print("\n[STEP 2/3] Training and benchmarking all 5 machine learning models...")
    metrics = train_and_evaluate_all()

    # Step 11-13: Generate plots and reports
    print("\n[STEP 3/3] Generating visual analytics and evaluation charts...")
    generate_evaluation_plots()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"TRAINING WORKFLOW COMPLETED SUCCESSFULLY IN {elapsed}s!")
    print(f"Selected Best Model: {metrics['best_model']} (Macro F1 = {metrics['best_macro_f1']:.4f})")
    print("Model artifacts saved to: ./models/")
    print("Evaluation charts saved to: ./results/plots/")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
