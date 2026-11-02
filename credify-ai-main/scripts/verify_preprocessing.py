"""
verify_preprocessing.py

This script verifies each preprocessing step of the CredifyAI pipeline.
Run this from the project root to validate data loading, schema enforcement,
cleaning, and target creation.
"""
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from pathlib import Path
from src.preprocess import (
    load_data,
    select_required_columns,
    clean_data,
    create_target_variable
)

RAW_DATA_PATH = Path("data/raw/credify-ai-dataset.csv")


def main():
    print("\nSTEP 1: Loading raw dataset")
    df = load_data(RAW_DATA_PATH)

    print("\nSTEP 2: Selecting required columns")
    df = select_required_columns(df)

    print("\nSTEP 3: Cleaning data")
    df = clean_data(df)

    print("\nSTEP 4: Creating target variable")
    df = create_target_variable(df)

    print("\nPreprocessing pipeline completed successfully.")
    print(f"Final dataset shape: {df.shape}")


if __name__ == "__main__":
    main()
