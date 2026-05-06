from churnguard.data import load_data, preprocess
from churnguard.evaluate import compute_metrics
from churnguard.train import train_model

DATA_PATH = "data/telco_churn.csv"


def test_train_model_returns_fitted_pipeline():
    df = load_data(DATA_PATH).head(300)
    X, y = preprocess(df)
    model = train_model(X, y, model_name="rf", params={"n_estimators": 5, "random_state": 42})
    predictions = model.predict(X.head(3))
    assert len(predictions) == 3


def test_compute_metrics_returns_expected_keys():
    df = load_data(DATA_PATH).head(500)
    X, y = preprocess(df)
    model = train_model(X, y, model_name="rf", params={"n_estimators": 5, "random_state": 42})
    metrics = compute_metrics(model, X, y)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
