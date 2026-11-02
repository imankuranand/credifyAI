"""
preprocess.py

This module is responsible for loading and preparing raw data
before feature engineering and model training.

At this stage, the only responsibility is:
- Safely loading the raw CSV dataset
- Verifying that the data has been loaded correctly
"""

import pandas as pd
from pathlib import Path

# Columns required for CredifyAI core preprocessing
# This acts as a schema contract for the dataset
REQUIRED_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "installment",
    "open_acc",
    "revol_util",
    "delinq_2yrs",
    "loan_status"   # used temporarily for target creation
]


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path object pointing to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.

    Why this function exists
    ------------------------
    - Isolates the data-loading logic
    - Makes the pipeline modular and debuggable
    - Ensures consistent loading behavior across environments
    """

    # --- Safety check: ensure the file actually exists ---
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")

    # --- Load the CSV file ---
    # low_memory=False ensures pandas infers data types in one pass
    # which avoids inconsistent dtype issues for large files
    dataframe = pd.read_csv(
        file_path,
        low_memory=False
    )

    # --- Logging for sanity check ---
    # Shape helps us quickly verify:
    # 1. The file loaded successfully
    # 2. The dataset size matches expectations
    print(f"Dataset loaded successfully.")
    print(f"Shape of dataset: {dataframe.shape}")

    # Return the loaded DataFrame for downstream processing
    return dataframe

def select_required_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Select and validate required columns from the raw dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw dataset loaded from the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the required columns.

    Raises
    ------
    ValueError
        If any required column is missing from the dataset.
    """

    # Convert column names to a set for efficient lookup
    available_columns = set(dataframe.columns)
    required_columns = set(REQUIRED_COLUMNS)

    # Identify missing columns (if any)
    missing_columns = required_columns - available_columns

    # If even one required column is missing, stop execution
    if missing_columns:
        raise ValueError(
            f"Missing required columns in dataset: {missing_columns}"
        )

    # Select only the approved columns (schema enforcement)
    filtered_dataframe = dataframe[REQUIRED_COLUMNS].copy()

    # Log shape for sanity check
    print(f"After column selection: {filtered_dataframe.shape}")

    return filtered_dataframe

def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and invalid financial data.

    Cleaning principles
    -------------------
    - Drop rows where core financial meaning is lost
    - Impute values only when a conservative, explainable assumption exists
    - Avoid aggressive row deletion to reduce bias
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = dataframe.copy()

    # -------------------------------
    # 1. Drop rows with fatal missing values
    # -------------------------------
    fatal_columns = ["annual_inc", "loan_amnt", "installment"]

    initial_rows = len(df)
    df = df.dropna(subset=fatal_columns)
    print(f"Dropped {initial_rows - len(df)} rows due to fatal missing values")

    # -------------------------------
    # 2. Remove invalid financial values
    # -------------------------------
    df = df[
        (df["annual_inc"] > 0) &
        (df["loan_amnt"] > 0) &
        (df["installment"] > 0)
    ]

    # -------------------------------
    # 3. Fix fixable numerical columns
    # -------------------------------

    # Delinquency: missing implies no reported delinquency
    df["delinq_2yrs"] = df["delinq_2yrs"].fillna(0)

    # Active accounts: fill with median to preserve distribution
    df["open_acc"] = df["open_acc"].fillna(df["open_acc"].median())

    # Debt-to-income: fill with median (safe central tendency)
    df["dti"] = df["dti"].fillna(df["dti"].median())

    # -------------------------------
    # 4. Clean revolving utilization
    # -------------------------------

    # Remove percentage sign if present and convert to float
    if df["revol_util"].dtype == "object":
        df["revol_util"] = df["revol_util"].str.rstrip("%")

    df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    # Drop rows where revol_util could not be converted
    df = df.dropna(subset=["revol_util"])

    print(f"After cleaning: {df.shape}")

    return df

def create_target_variable(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create the credit risk target variable from loan_status.

    Risk Mapping Strategy
    ---------------------
    Low Risk:
        - Fully Paid
        - Current

    Moderate Risk:
        - Late (16-30 days)
        - Late (31-120 days)
        - In Grace Period

    High Risk:
        - Charged Off
        - Default
    """

    df = dataframe.copy()

    # Define risk mapping
    risk_mapping = {
        "Fully Paid": "Low",
        "Current": "Low",
        "Late (16-30 days)": "Moderate",
        "Late (31-120 days)": "Moderate",
        "In Grace Period": "Moderate",
        "Charged Off": "High",
        "Default": "High"
    }

    # Keep only rows with known, meaningful outcomes
    df = df[df["loan_status"].isin(risk_mapping.keys())]

    # Create target column
    df["risk_level"] = df["loan_status"].map(risk_mapping)

    # Remove loan_status to prevent leakage
    df = df.drop(columns=["loan_status"])

    # Log class distribution
    print("Risk level distribution:")
    print(df["risk_level"].value_counts())

    return df

