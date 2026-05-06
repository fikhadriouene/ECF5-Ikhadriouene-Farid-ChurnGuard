# ChurnGuard MLOps — version cours

Projet ECF MLOps : industrialisation simple d'un modèle de prédiction de churn client.

Cette version reste volontairement proche des notions vues en cours :

- environnement virtuel `venv` ;
- dépendances dans `requirements.txt` ;
- code Python structuré en modules ;
- tests avec `pytest` et `pytest-cov` ;
- API avec FastAPI ;
- suivi d'expérience avec MLflow ;
- conteneurisation avec Docker ;
- orchestration simple avec docker compose ;
- CI GitHub Actions simple.

Les notions non vues en cours ne sont pas utilisées : pas de Poetry, pas de uv, pas de Ruff, pas de mypy, pas de pre-commit, pas de Trivy, pas de publication GHCR.

## Structure

```text
churnguard/
├── data.py        # chargement et préparation des données
├── train.py       # entraînement + log MLflow
└── evaluate.py    # calcul des métriques
api/
└── main.py        # API FastAPI
tests/
├── test_data.py
├── test_train.py
└── test_api.py
```

## Installation locale avec venv

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Lancer les tests

```bash
pytest --cov=churnguard --cov-fail-under=70
```

## Lancer MLflow localement

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Dans un autre terminal :

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m churnguard.train --model all --register
```

Ensuite, ouvrir : <http://127.0.0.1:5000>

Dans l'interface MLflow, promouvoir le meilleur modèle `churnguard` en `Production`.

## Lancer l'API localement

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uvicorn api.main:app --reload
```

Healthcheck :

```bash
curl http://127.0.0.1:8000/health
```

Exemple de prédiction :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

## Lancer avec Docker Compose

```bash
docker compose up --build
```

Services disponibles :

- API : <http://127.0.0.1:8000>
- MLflow : <http://127.0.0.1:5000>

## Schéma d'architecture

```text
Dataset CSV
   |
   v
churnguard.data -> churnguard.train -> MLflow Tracking / Registry
                                           |
                                           v
                                      FastAPI /predict
                                           |
                                           v
                                    Réponse JSON churn
```

## Justification pédagogique

Le sujet mentionne aussi Poetry/uv, Ruff, mypy, pre-commit, Trivy et GHCR. Ces outils sont utiles en entreprise, mais cette version les remplace par des outils vus en cours afin de produire une solution simple, compréhensible et défendable à l'oral.
