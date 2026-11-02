"""
features.py

This module handles feature engineering for the CredifyAI project.
It converts cleaned financial attributes into a compact, interpretable
set of features suitable for credit risk modeling.
"""

import pandas as pd
from pathlib import Path

def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready features from the cleaned dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset containing financial columns and the target variable.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataset ready for model training.
    """

    # Work on a copy to avoid mutating the original DataFrame
    df = dataframe.copy()

    # -------------------------------------------------
    # Feature 1: Debt-to-Income (already provided)
    # -------------------------------------------------
    # Represents overall debt stress from existing obligations
    df["dti"] = df["dti"]

    # -------------------------------------------------
    # Feature 2: Credit Utilization
    # -------------------------------------------------
    # Measures how much revolving credit is already being used
    df["credit_utilization"] = df["revol_util"]

    # -------------------------------------------------
    # Feature 3: EMI-to-Income Ratio
    # -------------------------------------------------
    # Shows monthly repayment burden relative to income
    monthly_income = df["annual_inc"] / 12
    df["emi_to_income"] = df["installment"] / monthly_income

    # -------------------------------------------------
    # Feature 4: Loan-to-Income Ratio
    # -------------------------------------------------
    # Shows size of loan relative to earning capacity
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]

    # -------------------------------------------------
    # Feature 5: Active Loan Count
    # -------------------------------------------------
    # Indicates financial complexity and repayment management load
    df["active_loan_count"] = df["open_acc"]

    # -------------------------------------------------
    # Feature 6: Delinquency Count
    # -------------------------------------------------
    # Past default-related behavior is one of the strongest predictors
    df["delinquency_count"] = df["delinq_2yrs"]

    # -------------------------------------------------
    # Final feature set + target
    # -------------------------------------------------
    feature_columns = [
        "dti",
        "credit_utilization",
        "emi_to_income",
        "loan_to_income",
        "active_loan_count",
        "delinquency_count",
        "risk_level"
    ]

    final_df = df[feature_columns]

    print(f"Final feature dataset shape: {final_df.shape}")

    return final_df


def save_features(dataframe: pd.DataFrame, output_path: Path) -> None:
    """
    Save the feature-engineered dataset to disk.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Final feature dataset.
    output_path : Path
        Path where the CSV file will be saved.
    """

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_path, index=False)

    print(f"Feature dataset saved to {output_path}")
