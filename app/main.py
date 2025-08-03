"""FastAPI application to serve MLP risk classification model."""
from pathlib import Path
import os
from typing import Optional

import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Directory containing model artifacts. Can be overridden via MODEL_DIR env var.
MODEL_DIR = Path(os.getenv("MODEL_DIR", "."))

app = FastAPI(title="MLP Risk Classifier API")

# Globals for the loaded artifacts
model: Optional[tf.keras.Model] = None
preprocessor = None
scaler = None
label_encoder = None


class PatientFeatures(BaseModel):
    """Input data for prediction."""
    sexo: str
    glosa_red: str
    edad_al_hospitalizarse: float
    plan_para_asegurados: str
    plan_catastrofico_asegurado: str
    estado_vigencia_cobertura: str
    prestacion_basica: str
    cto_dto: float


@app.on_event("startup")
def load_artifacts() -> None:
    """Load model and preprocessing artifacts into memory."""
    global model, preprocessor, scaler, label_encoder
    model_path = MODEL_DIR / "mlp_tf_model"
    pipeline_path = MODEL_DIR / "mlp_tf_pipeline.pkl"
    if not model_path.exists() or not pipeline_path.exists():
        raise RuntimeError(
            "Model or pipeline not found. Set MODEL_DIR to directory containing"
            " 'mlp_tf_model' and 'mlp_tf_pipeline.pkl'."
        )
    model = tf.keras.models.load_model(model_path)
    pipeline = joblib.load(pipeline_path)
    preprocessor = pipeline["preprocessor"]
    scaler = pipeline["scaler"]
    label_encoder = pipeline["label_encoder"]


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(features: PatientFeatures) -> dict[str, object]:
    """Predict risk level for a patient."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = pd.DataFrame([features.dict()])
    tX = preprocessor.transform(data)
    tX = scaler.transform(tX)
    probs = model.predict(tX)
    pred_idx = int(probs.argmax(axis=1)[0])
    label = label_encoder.inverse_transform([pred_idx])[0]
    return {"prediction": label, "probabilities": probs[0].tolist()}
