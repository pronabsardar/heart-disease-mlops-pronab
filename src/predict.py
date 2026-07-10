"""Prediction utility."""
import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(model, input_dict):
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].max()
    return int(pred), float(proba)


if __name__ == "__main__":
    model = load_model()
    sample = {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
    }
    pred, conf = predict(model, sample)
    print(f"Prediction: {pred} | Confidence: {conf:.3f}")