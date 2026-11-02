"""
train_model.py

This module is responsible for training the credit risk classification model
for the CredifyAI project.

Responsibilities:
- Load feature-engineered data
- Prepare features and target
- Handle class imbalance
- Train an XGBoost multiclass classifier

This module does NOT handle evaluation or visualization.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


# -------------------------------------------------
# Data Loading
# -------------------------------------------------

def load_features(file_path: Path) -> pd.DataFrame:
    """
    Load the feature-engineered dataset from disk.

    Parameters
    ----------
    file_path : Path
        Path to the processed features CSV.

    Returns
    -------
    pd.DataFrame
        Feature dataset ready for modeling.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found at: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded feature dataset with shape: {df.shape}")

    return df


# -------------------------------------------------
# Feature / Target Separation
# -------------------------------------------------

def split_features_target(dataframe: pd.DataFrame):
    """
    Separate features (X) and target (y).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature-engineered dataset.

    Returns
    -------
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target labels.
    """

    X = dataframe.drop(columns=["risk_level"])
    y = dataframe["risk_level"]

    return X, y


# -------------------------------------------------
# Target Encoding
# -------------------------------------------------

def encode_target(y: pd.Series):
    """
    Encode string risk labels into numeric values.

    This encoding is required because ML models operate on numeric targets.
    The encoder is returned so predictions can be mapped back to class names.

    Returns
    -------
    y_encoded : np.ndarray
        Encoded target labels.
    label_encoder : LabelEncoder
        Fitted encoder for inverse transformation.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Encoded class mapping:")
    for cls, idx in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"{cls} -> {idx}")

    return y_encoded, label_encoder


# -------------------------------------------------
# Train / Validation Split
# -------------------------------------------------

def train_validation_split(X, y_encoded):
    """
    Perform a stratified train-validation split.

    Stratification ensures minority risk classes are preserved
    in both training and validation sets.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    print("Train set shape:", X_train.shape)
    print("Validation set shape:", X_val.shape)

    return X_train, X_val, y_train, y_val


# -------------------------------------------------
# Class Weight Computation
# -------------------------------------------------

def compute_weights(y_train: np.ndarray):
    """
    Compute class weights to address class imbalance.

    Minority classes receive higher weights so that
    misclassifications are penalized more heavily.

    Returns
    -------
    dict
        Mapping from class index to weight.
    """

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight_mapping = dict(zip(classes, weights))

    print("Computed class weights:")
    for cls, weight in class_weight_mapping.items():
        print(f"Class {cls} -> weight {weight:.3f}")

    return class_weight_mapping


# -------------------------------------------------
# Model Training
# -------------------------------------------------

def train_xgboost(X_train, y_train, class_weights):
    """
    Train an XGBoost multiclass classifier using class-weighted loss.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Encoded training labels.
    class_weights : dict
        Mapping of class indices to weights.

    Returns
    -------
    model : XGBClassifier
        Trained XGBoost model.
    """

    # Convert class weights into per-sample weights
    sample_weights = np.array([class_weights[label] for label in y_train])

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    print("XGBoost training completed successfully")

    return model

def main():
    print("train_model.py is a library module. Use scripts/verify_train_model.py to run training.")


if __name__ == "__main__":
    main()
