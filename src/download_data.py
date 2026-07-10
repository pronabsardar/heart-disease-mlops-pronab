"""Download Heart Disease UCI dataset."""
import os
import pandas as pd
from urllib.request import urlretrieve

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
RAW_DIR = "data/raw"
FILE_PATH = os.path.join(RAW_DIR, "heart_disease.csv")


def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Downloading dataset from UCI repository...")
    urlretrieve(DATA_URL, os.path.join(RAW_DIR, "raw.data"))

    df = pd.read_csv(os.path.join(RAW_DIR, "raw.data"), header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(FILE_PATH, index=False)
    print(f"Dataset saved to {FILE_PATH} | Shape: {df.shape}")
    return df


if __name__ == "__main__":
    download_dataset()