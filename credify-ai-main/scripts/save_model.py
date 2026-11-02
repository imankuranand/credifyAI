"""
save_model.py

Trains the High-vs-Rest credit risk model and saves it
as a reusable artifact.
"""

import sys
from pathlib import Path
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.train_model import (
    load_features,
    split_features_target,
    encode_target,
    train_validation_split,
)

from src.model_io import save_model


def main():
    DATA_PATH = Path("data/processed/credify_features.csv")
    MODEL_PATH = Path("models/credify_high_risk_model.joblib")

    # Load data
    df = load_features(DATA_PATH)
    X, y = split_features_target(df)

    # Encode target
    y_encoded, _ = encode_target(y)

    # Binary target: High vs Rest
    y_binary = (y_encoded == 0).astype(int)

    X_train, _, y_train, _ = train_validation_split(X, y_binary)

    # Train binary model
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

    # Save model
    save_model(
        model=model,
        feature_names=list(X.columns),
        output_path=MODEL_PATH
    )


if __name__ == "__main__":
    main()
