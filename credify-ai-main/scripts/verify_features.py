"""
verify_features.py

Verification script for Phase 2 (Feature Engineering) of CredifyAI.

This script:
- Runs the full preprocessing + feature engineering pipeline
- Verifies feature columns
- Prints dataset shape and sample rows
- Acts as a sanity check before model training

Run from project root:
    python scripts/verify_features.py
"""

import sys
from pathlib import Path

# -------------------------------------------------
# Ensure project root is on Python path
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import (
    load_data,
    select_required_columns,
    clean_data,
    create_target_variable
)
from src.features import create_features

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credify-ai-dataset.csv"


EXPECTED_FEATURE_COLUMNS = [
    "dti",
    "credit_utilization",
    "emi_to_income",
    "loan_to_income",
    "active_loan_count",
    "delinquency_count",
    "risk_level"
]


def main():
    print("\nSTEP 1: Load raw dataset")
    df = load_data(RAW_DATA_PATH)

    print("\nSTEP 2: Preprocessing")
    df = select_required_columns(df)
    df = clean_data(df)
    df = create_target_variable(df)

    print("\nSTEP 3: Feature engineering")
    df = create_features(df)

    print("\nSTEP 4: Feature validation")

    # Check column contract
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            f"Feature columns mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COLUMNS}\n"
            f"Found:    {actual_columns}"
        )

    print("✔ Feature column schema verified")

    # Basic sanity checks
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values after feature engineering")

    print("✔ No missing values detected")

    print("\nSTEP 5: Dataset summary")
    print(f"Final dataset shape: {df.shape}")
    print("\nRisk level distribution:")
    print(df["risk_level"].value_counts())

    print("\nSample rows:")
    print(df.head())

    print("\n✅ Feature engineering verification PASSED")


if __name__ == "__main__":
    main()
