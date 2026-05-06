"""API FastAPI simple pour prédire le churn client."""

from contextlib import asynccontextmanager
import os
from typing import Any, AsyncIterator

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# app = FastAPI(title="ChurnGuard API")

MODEL_NAME = "churnguard"
# MODEL_URI = os.getenv("MODEL_URI", "models:/churnguard/Production")
MODEL_URI = os.getenv("MODEL_URI", "models:/churnguard@production")

model: Any = None
model_version = "production"


class Customer(BaseModel):
    """Données d'un client pour la prédiction."""

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# @app.on_event("startup")
# def load_model() -> None:
#     """Charge le modèle MLflow au démarrage de l'API."""
#     global model
#     try:
#         import mlflow.sklearn

#         model = mlflow.sklearn.load_model(MODEL_URI)
#     except Exception:
#         model = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modèle MLflow au démarrage de l'API."""
    global model

    try:
        print(f"Chargement du modèle depuis {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        print("Modèle chargé avec succès")
    except Exception as e:
        print(f"Erreur chargement modèle MLflow : {e}")
        model = None

    yield


app = FastAPI(title="ChurnGuard API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Vérifie que l'API répond correctement."""
    return {"status": "ok", "model": MODEL_NAME, "version": model_version}


def predict_one(customer: Customer) -> dict[str, float | bool]:
    """Retourne la prédiction pour un client."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    df = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {"churn": bool(prediction), "probability": probability}


@app.post("/predict")
def predict(customer: Customer) -> dict[str, float | bool]:
    """Prédit le churn pour un seul client."""
    return predict_one(customer)


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]) -> list[dict[str, float | bool]]:
    """Prédit le churn pour une liste de clients."""
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Le batch ne doit pas être vide")
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Le batch ne doit pas dépasser 100 clients")

    return [predict_one(customer) for customer in customers]
