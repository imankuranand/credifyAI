"""
evaluate.py

This module handles model evaluation for CredifyAI.
It contains reusable evaluation utilities independent of training.
"""

from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_val, y_val, label_encoder):
    """
    Evaluate trained model on validation data.
    """

    y_pred = model.predict(X_val)

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
