# 🫀 Heart Disease Prediction - MLOps Pipeline

**Author:** Pronab Sardar  
**Course:** AIMLCZG523 - MLOps Assignment 01  
**Institution:** BITS Pilani

## Overview
End-to-end MLOps solution predicting heart disease risk using the UCI Heart Disease dataset, deployed as a production-ready API with CI/CD, containerization, Kubernetes orchestration, and monitoring.

## Architecture
Data → Preprocessing → Training (MLflow) → FastAPI → Docker → GitHub Actions → Kubernetes → Prometheus/Grafana

## Quick Start

### Local Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py
python src/train.py
uvicorn src.api:app --reload

Docker

docker build -t heart-disease-api:v1 .
docker run -p 8000:8000 heart-disease-api:v1

Kubernetes (Helm)


helm install heart-api ./helm/heart-disease
minikube service heart-api-service

API Usage
POST /predict

json

{
  "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
}
Response:

json

{
  "prediction": 1,
  "confidence": 0.87,
  "risk_label": "High Risk"
}


Project Structure
heart-disease-mlops-pronab/
├── .github/workflows/       # CI/CD pipelines
├── data/                    # Dataset
├── notebooks/               # EDA & training notebooks
├── src/                     # Source code
├── tests/                   # Unit tests
├── models/                  # Trained models
├── docker/                  # Docker files
├── k8s/                     # Kubernetes manifests
├── helm/                    # Helm charts
├── monitoring/              # Prometheus & Grafana configs
├── screenshots/             # Documentation screenshots
├── report/                  # Final report
├── Dockerfile
├── requirements.txt
└── README.md