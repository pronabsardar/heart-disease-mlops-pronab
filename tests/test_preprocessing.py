"""Tests for data preprocessing module."""
import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preprocessing import build_preprocessor, FeatureEngineer


def test_preprocessor_builds():
    """Test that preprocessor can be built."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_feature_engineer():
    """Test the custom FeatureEngineer transformer."""
    fe = FeatureEngineer()
    sample = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
    }])
    transformed = fe.transform(sample)
    assert "age_chol_interaction" in transformed.columns
    assert "hr_age_ratio" in transformed.columns
    assert "age_group" in transformed.columns
    assert "chol_category" in transformed.columns


def test_preprocessor_transforms():
    """Test full preprocessing pipeline."""
    preprocessor = build_preprocessor()
    sample = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
    }] * 5)  # Duplicate for fitting
    preprocessor.fit(sample)
    transformed = preprocessor.transform(sample)
    assert transformed.shape[0] == 5