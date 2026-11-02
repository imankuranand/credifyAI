"""
verify_train_model.py

Verification script for Phase 3: Model Training & Evaluation.
Trains the model and evaluates it on a validation set.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.train_model import (
    load_features,
    split_features_target,
    encode_target,
    train_validation_split,
    compute_weights,
    train_xgboost
)

from src.evaluate import evaluate_model


def main():
    DATA_PATH = Path("data/processed/credify_features.csv")

    df = load_features(DATA_PATH)

    X, y = split_features_target(df)
    y_encoded, label_encoder = encode_target(y)

    X_train, X_val, y_train, y_val = train_validation_split(X, y_encoded)

    class_weights = compute_weights(y_train)

    model = train_xgboost(X_train, y_train, class_weights)

    evaluate_model(model, X_val, y_val, label_encoder)


if __name__ == "__main__":
    main()
