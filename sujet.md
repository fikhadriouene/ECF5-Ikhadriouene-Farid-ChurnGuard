ECF  MLOps — Concepteur Développeur en IA (RNCP 35288, bloc 5)
ChurnGuard MLOps
Industrialiser un modèle de prédiction de résiliation client
2 jours · individuel ·
✦
Contexte fictif
Vous êtes consultant freelance, mandaté par TelcoFr, un opérateur de télécommunications français de 1,2 million de clients. La direction marketing a fait développer en interne un modèle de prédiction de résiliation (churn) — il fonctionne sur le poste de la data scientist qui l'a écrit, mais personne d'autre ne sait le faire tourner. Le notebook est un mille-feuille de cellules, le modèle est un fichier .pkl envoyé par mail, et chaque appel à la prédiction passe par un copier-coller dans Jupyter.
Votre mission : reprendre ce projet et l'amener au standard MLOps de l'entreprise. À la fin des deux jours, n'importe quel développeur doit pouvoir cloner le repo, faire `docker compose up`, et obtenir une API de prédiction fonctionnelle. La CI doit tourner sur chaque push, le modèle doit être versionné dans MLflow, et les tests doivent passer.
Ce que vous ne faites pas
Vous ne touchez pas à la data science. Pas d'EDA, pas de feature engineering, pas de tuning d'hyperparamètres. Le modèle existant est ce qu'il est — votre rôle est de l'industrialiser, pas de l'améliorer. Si vous changez les performances du modèle, ce n'est pas un objectif évalué.
✦
Données
Le dataset est public : Telco Customer Churn, publié par IBM Sample Data sur la plateforme IBM Cognos Analytics. Il contient 7 043 clients d'un opérateur télécom américain et 21 colonnes (profil, abonnements, facturation, label de churn).
Source officielle :
— IBM Sample Data Sets — https://www.ibm.com/community/blogs/datasets/
— Mirror Kaggle (téléchargement direct) — https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Le fichier source CSV (utilisable sans authentification) est également hébergé par plusieurs tiers à des fins de reproduction académique. Le repo de départ contient un script `scripts/download_data.py` qui télécharge automatiquement le fichier depuis une URL stable et vérifie son intégrité par checksum SHA-256.
Caractéristique	Valeur
Nombre de lignes	7 043 clients
Nombre de colonnes	21 (incluant le label `Churn`)
Type de problème	Classification binaire (Churn = Yes / No)
Déséquilibre des classes	≈ 26,5 % de churners
Licence	IBM Sample Data — usage libre à des fins éducatives et de recherche
Format	CSV, ~960 Ko

Schéma des colonnes (pour information, vous n'avez pas à le manipuler vous-même) :
— Identité : customerID, gender, SeniorCitizen, Partner, Dependents.
— Ancienneté et abonnements : tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
— Contrat et facturation : Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
— Label : Churn (Yes / No).
✦
Repo de départ
Vous recevez un repository volontairement « sale » mais fonctionnel. Vous le clonez, vous l'amenez au standard de production. Aucune ligne du notebook ne doit être modifiée pour améliorer le modèle — vous extrayez son contenu en modules Python propres.
Contenu fourni :
— `notebook/exploration.ipynb` — le notebook de la data scientist (entraînement RandomForest, métriques, sauvegarde .pkl). Il marche.
— `scripts/download_data.py` — télécharge le CSV depuis la source publique avec vérification SHA-256.
— `README.md` — mission décrite en quelques lignes, sans solution.
— `.gitignore` Python standard.
— `requirements.txt` minimal (pandas, scikit-learn, jupyter).
Ce que le repo de départ ne contient pas (et que vous devez ajouter) :
— Pas de tests.
— Pas de structure modulaire (tout est dans le notebook).
— Pas de Dockerfile, pas de docker-compose.
— Pas de configuration MLflow.
— Pas d'API.
— Pas de CI.
✦
Etape 1 : Refactoring, tests, MLflow. À la fin du jour 1, votre repo est propre, vos tests passent et vous avez un modèle promu en production dans MLflow.
Étape 1.1 — Structure du projet
— Créer un package Python `churnguard/` avec les modules suivants :
churnguard/
├── __init__.py
├── data.py        # chargement, préprocessing, split
├── train.py       # entraînement, évaluation, persistance
└── evaluate.py    # métriques, rapport classification
tests/
├── test_data.py
└── test_train.py
— Initialiser un `pyproject.toml` (uv ou Poetry au choix) avec dépendances explicites.
— Configurer pre-commit hooks : ruff (lint + format), mypy (typage strict sur le package).
Étape 1.2 — Extraction du notebook vers les modules
— Copier le code du notebook dans les modules en respectant la séparation des responsabilités.
— Toute fonction publique doit avoir un type-hint et un docstring d'une ligne.
— `data.load_data(path: str) -> pd.DataFrame` charge le CSV brut.
— `data.preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]` fait le préprocessing et retourne X, y.
— `train.train_model(X, y, model_name: str, params: dict) -> Pipeline` entraîne et retourne le pipeline scikit-learn.
— `evaluate.compute_metrics(model, X_test, y_test) -> dict` retourne accuracy, precision, recall, f1, roc_auc.
Étape 1.3 — Tests pytest (6 tests minimum)
— Test 1 — `test_load_data_returns_dataframe` : vérifie la forme du DataFrame chargé.
— Test 2 — `test_load_data_has_expected_columns` : vérifie la présence des 21 colonnes attendues.
— Test 3 — `test_preprocess_returns_features_and_target` : X et y séparés correctement.
— Test 4 — `test_preprocess_handles_missing_total_charges` : la colonne TotalCharges contient des espaces vides à convertir — testez le cas.
— Test 5 — `test_train_model_returns_fitted_pipeline` : modèle entraîné, capable de prédire.
— Test 6 — `test_compute_metrics_returns_expected_keys` : dict avec les 5 métriques attendues.
— Coverage minimum : 70 % sur le package `churnguard`. Outil : pytest-cov.
Étape 1.4 — MLflow tracking
— Lancer un serveur MLflow local : `mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`
— Adapter `train.py` pour logger automatiquement chaque entraînement : paramètres, métriques (les 5), artefacts (modèle), signature, exemple d'input.
— Entraîner et logger 3 modèles : LogisticRegression, RandomForestClassifier, GradientBoostingClassifier — chacun dans un `mlflow.start_run()` avec un nom de run explicite.
— Comparer les 3 runs dans l'UI MLflow (http://127.0.0.1:5000).
Étape 1.5 — MLflow Model Registry
— Enregistrer le meilleur modèle dans le registry sous le nom `churnguard`.
— Promouvoir manuellement la version 1 en stage `Staging`, puis `Production` via l'UI ou via le SDK.
— Vérifier que vous pouvez charger le modèle promu : `mlflow.pyfunc.load_model("models:/churnguard/Production")`.
— Bonus : un script `python -m churnguard.train --model rf --register` qui enchaîne entraînement + log + registry.
Livrable 1 :
— Repo refactorisé, structure modulaire claire, tests verts (`pytest --cov`).
— 3 modèles loggés dans MLflow, 1 modèle promu en `Production` dans le registry.
— Capture d'écran de l'UI MLflow (à inclure dans le README final).
✦
Etape 2 : API, Docker, GitHub Actions. À la fin du jour 2, `docker compose up` suffit à démarrer toute la stack et la CI tourne sur chaque push.
Étape 2.1 — API FastAPI
— Créer un module `api/main.py` avec FastAPI.
— `GET /health` retourne `{"status": "ok", "model": "churnguard", "version": "<version MLflow>"}`.
— `POST /predict` reçoit un dict des features d'un client, retourne `{"churn": true|false, "probability": 0.0–1.0}`.
— `POST /predict/batch` reçoit une liste de clients (max 100), retourne la liste des prédictions.
— Validation Pydantic v2 stricte sur les types et bornes raisonnables (tenure ≥ 0, MonthlyCharges ≥ 0, etc.).
— Le modèle est chargé une seule fois au démarrage via `lifespan` depuis MLflow registry (`models:/churnguard/Production`).
— Gestion d'erreurs HTTP : 422 sur payload invalide, 400 sur batch vide ou > 100, 503 si modèle non chargé.
Étape 2.2 — Dockerfile multi-stage
— Stage `builder` : python:3.11-slim, installe les dépendances dans un venv.
— Stage `runtime` : python:3.11-slim, copie le venv et le code, expose le port 8000, lance uvicorn.
— Image finale visée : moins de 500 Mo. Vérifier avec `docker images`.
— Bonus : utilisateur non-root, healthcheck Docker, .dockerignore propre.
Étape 2.3 — docker-compose
— Service `mlflow` : serveur MLflow avec backend SQLite et volume persistant pour `./mlruns`.
— Service `api` : votre API FastAPI, dépend de `mlflow`, variable d'environnement `MLFLOW_TRACKING_URI` pointant vers `mlflow:5000`.
— Réseau Docker partagé entre les deux services.
— Test manuel : `docker compose up --build` puis `curl http://localhost:8000/health`, puis `curl -X POST http://localhost:8000/predict -d '{...}'`.
Étape 2.4 — Workflow CI
— Fichier `.github/workflows/ci.yml` déclenché sur `push` et `pull_request`.
— Job `lint` : ruff check + ruff format --check.
— Job `typecheck` : mypy sur le package.
— Job `test` : pytest avec coverage, échec si coverage < 70 %.
— Job `build` : docker build de l'image API, scan Trivy sur l'image.
— Tous les jobs en parallèle quand c'est possible (`needs:` uniquement quand nécessaire).
— Cache pip / uv pour accélérer les exécutions répétées.
Étape 2.5 — Workflow CD (sur tag)
— Fichier `.github/workflows/release.yml` déclenché sur `push` de tag `v*.*.*`.
— Build de l'image API + push vers GitHub Container Registry (ghcr.io/<user>/churnguard:tag).
— Login via `${{ secrets.GITHUB_TOKEN }}` (pas de secret à configurer manuellement).
— Bonus : génération automatique d'un release notes via Conventional Commits.
Étape 2.6 — README et démo
— README final "vitrine" : badge CI, schéma d'architecture (ASCII ou image), section quickstart `docker compose up`, exemple d'appel API avec curl, lien vers l'image GHCR.
— Démo individuelle 15 minutes : montrer le pipeline de bout en bout (clone → docker compose up → curl predict → modification de code → push → CI verte).
Livrable final :
— Repo public sur GitHub, badge CI vert.
— Image Docker de l'API publiée sur ghcr.io ou un autre registry.
— `docker compose up --build` démarre toute la stack en moins de 2 minutes.
— README à jour, démo prête.
✦

Bonus  :
— Monitoring de drift avec Evidently sur un échantillon de production simulé.
— Healthcheck Docker actif et utilisé dans docker-compose.
— Image Docker < 300 Mo (multi-stage agressif, distroless ou alpine).
— Déploiement final sur un cluster k3s local (kind, k3d) avec manifests YAML.
— Notification Slack ou email sur échec de CI.
✦
Bon ECF.