"""FastAPI application for Heart Disease Prediction with Monitoring."""
import logging
import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTION_COUNT = Counter("predictions_total", "Prediction count by class", ["prediction"])
PREDICTION_CONFIDENCE = Histogram("prediction_confidence", "Model confidence distribution")
MODEL_FEATURE_MEAN = Gauge("input_feature_mean", "Mean of input features", ["feature"])

# Global model reference
MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global MODEL
    try:
        MODEL = joblib.load("models/best_model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL = None
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment 01 - Pronab Sardar",
    version="1.0.0",
    lifespan=lifespan
)


class HeartInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1=Male, 0=Female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak ST segment")
    ca: float = Field(..., ge=0, le=4, description="Major vessels colored")
    thal: float = Field(..., ge=0, le=3, description="Thalassemia")


class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    risk_label: str


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "author": "Pronab Sardar",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: HeartInput):
    start = time.time()
    if MODEL is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([payload.model_dump()])
        pred = int(MODEL.predict(df)[0])
        proba = float(MODEL.predict_proba(df)[0].max())
        label = "High Risk" if pred == 1 else "Low Risk"

        # Metrics
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
        PREDICTION_COUNT.labels(prediction=str(pred)).inc()
        PREDICTION_CONFIDENCE.observe(proba)

        # Feature drift monitoring
        for feature, value in payload.model_dump().items():
            MODEL_FEATURE_MEAN.labels(feature=feature).set(value)

        logger.info(f"Prediction: {pred} | Confidence: {proba:.3f} | Label: {label}")
        return PredictionOutput(prediction=pred, confidence=proba, risk_label=label)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)