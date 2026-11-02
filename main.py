"""FastAPI service that serves the churn model."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import CustomerData, HealthResponse, PredictionResponse
from src.config import load_config

state = {"model": None, "config": load_config()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    state["config"] = cfg
    path = Path(cfg["model_path"])
    state["model"] = joblib.load(path) if path.exists() else None
    yield
    state["model"] = None


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API. See /docs, /health, /predict."}


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = state["model"] is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                          model_version=state["config"]["model_version"])


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    frame = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(frame)[0, 1])
    threshold = float(state["config"]["decision_threshold"])
    return PredictionResponse(
        prediction="Churn" if proba >= threshold else "Not Churn",
        churn_probability=round(proba, 4),
        threshold=threshold,
        model_version=state["config"]["model_version"])
