"""
test_inference.py

Sanity test for CredifyAI inference pipeline.
"""

import sys
from pathlib import Path

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import predict_risk


def main():
    sample_input = {
        "dti": 22.0,
        "credit_utilization": 65.0,
        "emi_to_income": 0.45,
        "loan_to_income": 2.8,
        "active_loan_count": 3,
        "delinquency_count": 1
    }

    result = predict_risk(sample_input)
    print(result)


if __name__ == "__main__":
    main()
