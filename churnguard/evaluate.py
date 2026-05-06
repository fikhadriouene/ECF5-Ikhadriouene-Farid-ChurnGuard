"""Fonctions d'évaluation du modèle."""

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(model: Any, X_test: Any, y_test: Any) -> dict[str, float]:
    """Calcule les principales métriques de classification."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
