"""Tests for trained model."""
import sys
import os
import joblib
import pandas as pd
import pytest

# Ensure src is importable BEFORE loading the pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import to register the FeatureEngineer class (needed for pickle deserialization)
from src.data_preprocessing import FeatureEngineer  # noqa: F401


@pytest.mark.skipif(not os.path.exists("models/best_model.pkl"),
                    reason="Model not trained yet")
def test_model_predicts():
    """Test that the model produces valid predictions."""
    model = joblib.load("models/best_model.pkl")
    sample = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
    }])
    pred = model.predict(sample)
    proba = model.predict_proba(sample)
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert 0 <= proba[0].max() <= 1


@pytest.mark.skipif(not os.path.exists("models/best_model.pkl"),
                    reason="Model not trained yet")
def test_model_batch_predictions():
    """Test model handles batch inputs."""
    model = joblib.load("models/best_model.pkl")
    sample = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
    }] * 3)
    preds = model.predict(sample)
    assert len(preds) == 3