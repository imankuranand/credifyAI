"""
inference.py

Inference utilities for the CredifyAI project.

This module handles:
- Loading the trained model
- Validating incoming feature inputs
- Generating risk predictions for new borrowers

No training logic exists here.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.risk_reasons import generate_risk_reasons, generate_insights, generate_simulations
from src.model_io import load_model


# -------------------------------------------------
# Constants
# -------------------------------------------------

MODEL_PATH = Path("models/credify_high_risk_model.joblib")
DEFAULT_THRESHOLD = 0.5


# -------------------------------------------------
# Inference Utilities
# -------------------------------------------------

def load_inference_model(model_path: Path = MODEL_PATH):
    """
    Load trained model and feature schema for inference.
    """

    model, feature_names = load_model(model_path)
    return model, feature_names


def prepare_input(features: dict, feature_names: list) -> pd.DataFrame:
    """
    Validate and align raw input features with training schema.

    Parameters
    ----------
    features : dict
        Dictionary containing borrower feature values.
    feature_names : list
        Ordered list of features expected by the model.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame aligned with model input.
    """

    missing = set(feature_names) - set(features.keys())
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    ordered_values = {f: features[f] for f in feature_names}
    return pd.DataFrame([ordered_values])

def predict_risk(
    features: dict
):
    """
    Predict credit risk with industry-grade interpretation.

    Returns:
    - probability of High Risk
    - risk band (Low / Medium / High)
    - decision guidance
    """

    model, feature_names = load_inference_model()
    X_input = prepare_input(features, feature_names)

    prob_high_risk = model.predict_proba(X_input)[0, 1]

    # -----------------------------
    # Risk Band Policy Layer
    # -----------------------------
    if prob_high_risk < 0.05:
        risk_band = "Low Risk"
        decision = "Eligible for approval"
    elif prob_high_risk < 0.15:
        risk_band = "Medium Risk (Watchlist)"
        decision = "Requires manual review"
    else:
        risk_band = "High Risk"
        decision = "High likelihood of default"

    reasons = generate_risk_reasons(features)
    insights = generate_insights(features, prob_high_risk)
    simulations = generate_simulations(features, prob_high_risk, risk_band)

    return {
        "probability_high_risk": round(float(prob_high_risk), 4),
        "risk_band": risk_band,
        "decision": decision,
        "key_reasons": reasons,
        "insights": insights,
        "simulations": simulations
    }
