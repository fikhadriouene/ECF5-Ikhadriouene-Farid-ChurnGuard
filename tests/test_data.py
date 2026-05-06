import pandas as pd

from churnguard.data import EXPECTED_COLUMNS, load_data, preprocess

DATA_PATH = "data/telco_churn.csv"


def test_load_data_returns_dataframe():
    df = load_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 7043


def test_load_data_has_expected_columns():
    df = load_data(DATA_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_preprocess_returns_features_and_target():
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert len(X) == len(y)


def test_preprocess_handles_missing_total_charges():
    df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 2],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Electronic check", "Electronic check"],
            "MonthlyCharges": [29.85, 53.85],
            "TotalCharges": [" ", "108.15"],
            "Churn": ["No", "Yes"],
        }
    )
    X, y = preprocess(df)
    assert len(X) == 1
    assert len(y) == 1
