"""
verify_explainability.py

Phase 4 Verification Script: Model Explainability using SHAP.

This script generates stable, interpretable explanations for the
CredifyAI credit risk model.

Explainability Strategy
-----------------------
- Uses a One-vs-Rest (OvR) binary classification setup:
      High Risk (1) vs Not High Risk (0)
- Explains the model's predicted probability of High Risk
- Uses SHAP's unified Explainer API for compatibility with
  modern XGBoost versions

Outputs
-------
- Global SHAP summary plot identifying the most influential
  drivers of high-risk predictions
- Plot is saved under the `visuals/` directory
"""

import sys
from pathlib import Path

import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# -------------------------------------------------
# Project root & imports
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.train_model import (
    load_features,
    split_features_target,
    encode_target,
    train_validation_split,
)

from src.explain import init_shap_explainer, compute_shap_values


def main():
    """
    Run SHAP-based explainability for the High-vs-Rest credit risk model.
    """

    DATA_PATH = Path("data/processed/credify_features.csv")
    VISUALS_DIR = PROJECT_ROOT / "visuals"
    VISUALS_DIR.mkdir(exist_ok=True)

    # -------------------------------------------------
    # Load feature-engineered dataset
    # -------------------------------------------------

    df = load_features(DATA_PATH)
    X, y = split_features_target(df)

    # Encode multiclass labels
    y_encoded, label_encoder = encode_target(y)

    # -------------------------------------------------
    # Create binary target: High Risk vs Rest
    #
    # Label mapping from previous phases:
    #   High -> 0
    #   Low -> 1
    #   Moderate -> 2
    #
    # After conversion:
    #   1 -> High Risk
    #   0 -> Not High Risk
    # -------------------------------------------------

    y_binary = (y_encoded == 0).astype(int)

    # Stratified split preserves high-risk proportion
    X_train, X_val, y_train, _ = train_validation_split(X, y_binary)

    # -------------------------------------------------
    # Train binary XGBoost model
    # -------------------------------------------------

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("Binary High-vs-Rest model trained successfully")

    # -------------------------------------------------
    # SHAP Explainability
    # -------------------------------------------------

    # Small, representative samples for performance reasons
    background = X_train.sample(n=500, random_state=42)
    shap_sample = X_val.sample(n=1000, random_state=42)

    explainer = init_shap_explainer(model, background)
    shap_values = compute_shap_values(explainer, shap_sample)

    print("SHAP values computed successfully")

    # -------------------------------------------------
    # Global SHAP Summary Plot
    #
    # Index 1 corresponds to the probability of High Risk
    # when using predict_proba
    # -------------------------------------------------

    shap.summary_plot(
        shap_values.values[..., 1],
        shap_sample,
        show=False
    )

    output_path = VISUALS_DIR / "shap_summary_high_risk.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"SHAP summary plot saved to: {output_path}")


if __name__ == "__main__":
    main()
