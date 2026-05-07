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
ECF5-Ikhadriouene-Farid-ChurnGuard/
│
├── churnguard/                                 # Package principal MLOps
│   │
│   ├── __init__.py                             # Initialisation du package
│   │
│   ├── data.py                                 # Chargement et preprocessing des données
│   │                                           # - load_data()
│   │                                           # - preprocess()
│   │
│   ├── evaluate.py                             # Calcul des métriques ML
│   │                                           # - accuracy
│   │                                           # - precision
│   │                                           # - recall
│   │                                           # - f1-score
│   │                                           # - roc_auc
│   │
│   └── train.py                                # Entraînement des modèles
│                                               # Logging MLflow
│                                               # Registry & promotion
│
├── api/                                        # API REST FastAPI
│   │
│   └── main.py                                 # Endpoints :
│                                               # - GET /health
│                                               # - POST /predict
│                                               # - POST /predict/batch
│
├── scripts/                                    # Scripts utilitaires
│   │
│   └── download_data.py                        # Téléchargement automatique
│                                               # du dataset Telco Churn
│
├── tests/                                      # Tests automatisés pytest
│   │
│   ├── test_api.py                             # Tests API FastAPI
│   │
│   ├── test_data.py                            # Tests preprocessing
│   │                                           # et validation des données
│   │
│   └── test_train.py                           # Tests entraînement
│                                               # et calcul des métriques
│
├── data/                                       # Données locales
│   │
│   ├── .gitkeep                                # Conservation du dossier vide
│   │
│   └── telco_churn.csv                         # Dataset téléchargé
│                                               # automatiquement
│
├── notebook/                                   # Notebook initial fourni
│   │
│   └── exploration.ipynb                       # Notebook de départ
│                                               # de la data scientist
│
├── mlruns/                                     # Artefacts MLflow
│   │
│   ├── experiments/                            # Expérimentations ML
│   ├── metrics/                                # Métriques enregistrées
│   ├── models/                                 # Modèles versionnés
│   └── artifacts/                              # Artefacts MLflow
│
├── .github/                                    # GitHub Actions
│   │
│   └── workflows/
│       │
│       ├── ci.yml                              # Pipeline CI :
│       │                                       # tests + coverage + build
│       │
│       └── release.yml                         # Pipeline CD :
│                                               # build + push GHCR
│
├── Dockerfile                                  # Image Docker multi-stage
│
├── docker-compose.yml                          # Stack complète :
│                                               # MLflow + API + trainer
│
├── requirements.txt                            # Dépendances runtime
│
├── requirements-dev.txt                        # Dépendances dev/tests
│
├── .dockerignore                               # Optimisation du build Docker
│
├── .gitignore                                  # Exclusions Git
│
├── README.md                                   # Documentation du projet
│
├── sujet.md                                    # Sujet officiel ECF
│
└── pyproject.toml                              # Configuration Python/outillage

```

## Installation locale avec venv

```bash
python -m venv venv
# source venv/bin/activate   # Linux / Mac
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Lancer les tests

```bash
pytest --cov=churnguard --cov-fail-under=70
```

## Lancer MLflow localement

Lancement du serveur mlflow
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Dans un autre terminal :

Création de la variable d'environnement MLFLOW_TRACKING_URI et lancement de l'entrainement des modèles
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m churnguard.train --model all --register
```

Ensuite, ouvrir : <http://127.0.0.1:5000>

Dans l'interface MLflow, promouvoir le meilleur modèle `churnguard` en `Production`.

## Lancer l'API localement

Création de la variable d'environnement MLFLOW_TRACKING_URI et lancement de l'api
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uvicorn api.main:app --reload
```

Healthcheck :

```bash
curl http://127.0.0.1:8000/health


Réponse :

StatusCode        : 200
StatusDescription : OK
Content           : {"status":"ok","model":"churnguard","version":"production"}
RawContent        : HTTP/1.1 200 OK
                    Content-Length: 59
                    Content-Type: application/json
                    Date: Thu, 07 May 2026 10:09:00 GMT
                    Server: uvicorn

                    {"status":"ok","model":"churnguard","version":"production"}
Forms             : {}
Headers           : {[Content-Length, 59], [Content-Type, application/json], [Date, Thu, 07 May 2026 10:09:00 GMT],
                    [Server, uvicorn]}
Images            : {}
InputFields       : {}
Links             : {}
ParsedHtml        : mshtml.HTMLDocumentClass
RawContentLength  : 59

```

Exemple de prédiction simple :

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
Réponse :
{"churn":true,"probability":0.6226508903195026}


Exemple de prédictions multiples :

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
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
    },
    {
      "gender": "Male",
      "SeniorCitizen": 1,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 60,
      "PhoneService": "Yes",
      "MultipleLines": "Yes",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "Yes",
      "DeviceProtection": "Yes",
      "TechSupport": "Yes",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes",
      "Contract": "Two year",
      "PaperlessBilling": "No",
      "PaymentMethod": "Bank transfer (automatic)",
      "MonthlyCharges": 110.5,
      "TotalCharges": 6500.0
    }
  ]'
  ```
  Réponse :
  [{"churn":true,"probability":0.6226508903195026},{"churn":false,"probability":0.0610259370490438}]


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


