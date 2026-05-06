"""Fonctions de chargement et de préparation des données churn."""

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str) -> pd.DataFrame:
    """Charge le fichier CSV brut dans un DataFrame pandas."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prépare les données et sépare les variables X et la cible y."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Découpe les données en jeu d'entraînement et jeu de test."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
