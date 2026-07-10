"""Enhanced preprocessing with feature engineering."""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak",
                    "age_chol_interaction", "hr_age_ratio", "bp_chol_ratio"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
                        "age_group", "chol_category"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Interaction features
        X["age_chol_interaction"] = X["age"] * X["chol"] / 1000
        X["hr_age_ratio"] = X["thalach"] / X["age"]
        X["bp_chol_ratio"] = X["trestbps"] / X["chol"]

        # Age binning
        X["age_group"] = pd.cut(X["age"], bins=[0, 40, 55, 70, 120],
                                labels=[0, 1, 2, 3]).astype(int)

        # Cholesterol category
        X["chol_category"] = pd.cut(X["chol"], bins=[0, 200, 240, 600],
                                    labels=[0, 1, 2]).astype(int)
        return X


def build_preprocessor():
    """Build complete preprocessing pipeline with feature engineering."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    column_transformer = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    full_pipeline = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("column_transformer", column_transformer)
    ])
    return full_pipeline


def load_and_split(csv_path="data/raw/heart_disease.csv", test_size=0.2, random_state=42):
    """Load dataset and perform stratified train/test split."""
    df = pd.read_csv(csv_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_preprocessor(preprocessor, path="models/preprocessor.pkl"):
    """Save fitted preprocessor to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    save_preprocessor(preprocessor)
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")