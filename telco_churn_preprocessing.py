"""Preprocessing for scoring new Telco customers with the saved churn model."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NOMINAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaymentMethod",
]
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}


class TelcoChurnPreprocessor(BaseEstimator, TransformerMixin):
    """Clean, engineer, and encode raw Telco customer rows.

    Expects the same columns as TelcoCustomerChurn.csv. `customerID` and
    `Churn` are dropped when present. Columns are locked to the training
    schema so a new category does not change the feature layout.
    """

    def fit(self, X, y=None):
        self.feature_columns_ = list(self._engineer(X).columns)
        return self

    def transform(self, X):
        transformed = self._engineer(X)
        return transformed.reindex(columns=self.feature_columns_, fill_value=0)

    def _engineer(self, X):
        df = X.copy()
        df = df.drop(columns=[c for c in ("customerID", "Churn") if c in df.columns])

        df["TotalCharges"] = (
            pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype("float64")
        )
        df["ServiceCount"] = df[SERVICE_COLS].eq("Yes").sum(axis=1).astype("int64")
        df["IsAutoPay"] = (
            df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype("int64")
        )
        df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

        for col in YES_NO_COLS:
            df[col] = df[col].map({"Yes": 1, "No": 0}).astype("int64")
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1}).astype("int64")
        df["Contract"] = df["Contract"].map(CONTRACT_MAP).astype("int64")
        return pd.get_dummies(df, columns=NOMINAL_COLS, dtype="int64")
