"""
explain.py

Model explainability utilities for CredifyAI using SHAP.

This module uses SHAP's unified Explainer API and explicitly explains
model prediction probabilities, which is required for compatibility
with scikit-learn style models such as XGBClassifier.
"""

import shap
import pandas as pd


# -------------------------------------------------
# SHAP Explainer Initialization
# -------------------------------------------------

def init_shap_explainer(model, background_data: pd.DataFrame):
    """
    Initialize a SHAP explainer for a binary XGBoost classifier.

    Parameters
    ----------
    model
        Trained XGBClassifier.
    background_data : pd.DataFrame
        Representative background data.

    Returns
    -------
    shap.Explainer
        Initialized SHAP explainer.

    Notes
    -----
    - We explain `predict_proba`, not the raw model.
    - SHAP will explain the probability of the positive class.
    """

    explainer = shap.Explainer(
        model.predict_proba,
        background_data
    )
    return explainer


# -------------------------------------------------
# SHAP Value Computation
# -------------------------------------------------

def compute_shap_values(explainer, data: pd.DataFrame):
    """
    Compute SHAP values for the given dataset.

    Parameters
    ----------
    explainer : shap.Explainer
        Initialized explainer.
    data : pd.DataFrame
        Samples to explain.

    Returns
    -------
    shap.Explanation
        SHAP explanation object.

    Notes
    -----
    - For binary classification with predict_proba:
        shap_values.values has shape:
        (n_samples, n_features, 2)
    - Index 1 corresponds to the positive class (High Risk).
    """

    shap_values = explainer(data)
    return shap_values
