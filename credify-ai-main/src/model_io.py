"""
model_io.py

Utilities for saving and loading trained models and metadata.
"""

from pathlib import Path
from joblib import dump, load


def save_model(model, feature_names, output_path: Path):
    """
    Save trained model and feature metadata to disk.

    Parameters
    ----------
    model
        Trained ML model.
    feature_names : list
        Ordered list of feature names used during training.
    output_path : Path
        File path to save the model artifact.
    """

    artifact = {
        "model": model,
        "features": feature_names
    }

    dump(artifact, output_path)
    print(f"Model saved to {output_path}")


def load_model(model_path: Path):
    """
    Load trained model and metadata from disk.
    """

    artifact = load(model_path)
    return artifact["model"], artifact["features"]
