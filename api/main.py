
"""API FastAPI simple pour prédire le churn client."""

from contextlib import asynccontextmanager
import os
from typing import Any, AsyncIterator

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = "churnguard"
MODEL_URI = os.getenv("MODEL_URI", "models:/churnguard/production")

model: Any = None
model_version = "production"


class CustomerRequest(BaseModel):
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


class CustomerResponse(BaseModel):
    """Résultat de la prédiction de churn."""

    churn: bool
    probability: float


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


def predict_one(customer: CustomerRequest) -> CustomerResponse:
    """Retourne la prédiction pour un client."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    df = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return CustomerResponse(churn=bool(prediction), probability=probability)


@app.post("/predict", response_model=CustomerResponse)
def predict(customer: CustomerRequest) -> CustomerResponse:
    """Prédit le churn pour un seul client."""
    return predict_one(customer)


@app.post("/predict/batch", response_model=list[CustomerResponse])
def predict_batch(customers: list[CustomerRequest]) -> list[CustomerResponse]:
    """Prédit le churn pour une liste de clients."""
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Le batch ne doit pas être vide")
    if len(customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Le batch ne doit pas dépasser 100 clients",
        )

    return [predict_one(customer) for customer in customers]