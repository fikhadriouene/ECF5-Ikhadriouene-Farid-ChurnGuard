"""Entraînement simple des modèles avec scikit-learn et MLflow."""

import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.tracking import MlflowClient

from churnguard.data import NUMERIC_COLUMNS, load_data, preprocess, split_data
from churnguard.evaluate import compute_metrics
import os


MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "gb": GradientBoostingClassifier(random_state=42),
}


def train_model(X: pd.DataFrame, y: pd.Series, model_name: str = "rf", params: dict | None = None) -> Pipeline:
    """Entraîne un pipeline scikit-learn et retourne le modèle entraîné."""
    if model_name not in MODELS:
        raise ValueError("model_name doit être 'logreg', 'rf' ou 'gb'.")

    params = params or {}
    model = MODELS[model_name]
    model.set_params(**params)

    numeric_columns = [col for col in NUMERIC_COLUMNS if col in X.columns]
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


def train_and_log(data_path: str, model_name: str, register: bool = False) -> dict[str, float]:  # pragma: no cover
    """Entraîne un modèle, le log dans MLflow et retourne les métriques."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    import os

    df = load_data(data_path)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # mlflow.set_tracking_uri("http://127.0.0.1:5000")


    mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
    
    
    mlflow.set_experiment("churn-guard")

    with mlflow.start_run(run_name=model_name):
        model = train_model(X_train, y_train, model_name=model_name)
        metrics = compute_metrics(model, X_test, y_test)

        mlflow.log_param("model_name", model_name)
        mlflow.set_tag("algorithm", model_name)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test.head(5), model.predict(X_test.head(5)))
        input_example = X_test.head(1)

        # if register:
        #     mlflow.sklearn.log_model(
        #         model,
        #         artifact_path=f"churnguard_{model_name}",
        #         registered_model_name=f"churnguard",
        #         signature=signature,
        #         input_example=input_example,
        #     )
        # else:
        #     mlflow.sklearn.log_model(
        #         model,
        #         artifact_path=f"churnguard_{model_name}",
        #         signature=signature,
        #         input_example=input_example,
        #     )

        artifact_name = f"churnguard_{model_name}"
        
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_name,
            signature=signature,
            input_example=input_example,
        )

        if register:
            mlflow.register_model(
                model_uri=model_info.model_uri,
                name="churnguard",
            )

    return metrics


# def promote_best_model(model_name: str) -> None:
#     """Promote best model version to Production."""

#     client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

#     versions = client.search_model_versions(f"name='{model_name}'")

#     best_version = None
#     best_accuracy = 0.0

#     for version in versions:
#         run = client.get_run(version.run_id)

#         accuracy = run.data.metrics.get("test_accuracy", 0)

#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_version = version

#     if best_version:
#         client.transition_model_version_stage(
#             name=model_name,
#             version=best_version.version,
#             stage="Production",
#             archive_existing_versions=True,
#         )

#         print(
#             f"\nVersion {best_version.version} "
#             f"promue en Production "
#             f"(accuracy: {best_accuracy:.4f})"
#         )

def promote_best_model(model_name: str) -> None:
    """Promote best model version with alias production."""

    # client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

    versions = client.search_model_versions(f"name='{model_name}'")

    best_version = None
    best_accuracy = 0.0

    for version in versions:
        run = client.get_run(version.run_id)
        accuracy = run.data.metrics.get("accuracy", 0)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_version = version

    if best_version:
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=best_version.version,
        )

        print(
            f"Version {best_version.version} promue avec l'alias production "
            f"(accuracy: {best_accuracy:.4f})"
        )


def main() -> None:  # pragma: no cover
    """Lance l'entraînement depuis la ligne de commande."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco_churn.csv")
    parser.add_argument("--model", choices=["logreg", "rf", "gb", "all"], default="rf")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    models_to_train = ["logreg", "rf", "gb"] if args.model == "all" else [args.model]

    for model_name in models_to_train:
        metrics = train_and_log(args.data, model_name, register=args.register)
        print(model_name, metrics)
    print("promotion")
    promote_best_model("churnguard")

if __name__ == "__main__":  # pragma: no cover
    main()
