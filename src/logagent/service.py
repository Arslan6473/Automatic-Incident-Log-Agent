"""Early-warning API: first log events of an HDFS block -> calibrated failure probability + alert decision.

Run:  uv run uvicorn logagent.service:app --port 8000
"""

import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field, field_validator

from logagent.features import EARLY_FEATURES, EARLY_K, EVENT_IDS, early_vector

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
CONFIG = json.loads((MODEL_DIR / "serving_config.json").read_text())
if CONFIG["features"] != EARLY_FEATURES:
    raise RuntimeError("serving_config.json was exported with a different feature list than logagent.features")

SESSION = ort.InferenceSession(str(MODEL_DIR / "hdfs_lgbm.onnx"), providers=["CPUExecutionProvider"])
CAL_A, CAL_B = CONFIG["calibration"]["a"], CONFIG["calibration"]["b"]
THRESHOLD = CONFIG["threshold"]
MODEL_VERSION = f'{CONFIG["model_name"]}/v{CONFIG["model_version"]}'
_VALID_EVENTS = set(EVENT_IDS)


class BlockEvents(BaseModel):
    events: list[str] = Field(min_length=1, max_length=1000,
                              description=f"Event IDs (E1..E29) in log order; only the first {EARLY_K} are used")
    elapsed_seconds: float = Field(ge=0, le=7 * 24 * 3600,
                                   description=f"Seconds from the first to the {EARLY_K}th event (or last, if fewer)")

    @field_validator("events")
    @classmethod
    def known_events(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - _VALID_EVENTS)
        if unknown:
            raise ValueError(f"unknown event ids: {unknown[:5]}")
        return v


class Prediction(BaseModel):
    p_anomaly: float
    alert: bool
    threshold: float
    model_version: str


def predict_proba(x: np.ndarray) -> np.ndarray:
    p_raw = SESSION.run(["probabilities"], {"input": x})[0][:, 1].astype(np.float64)
    p_raw = np.clip(p_raw, 1e-12, 1 - 1e-12)
    return 1 / (1 + np.exp(CAL_A * np.log(p_raw / (1 - p_raw)) + CAL_B))


# Pre-filled in the /docs "Try it out" form (dropdown), taken from real HDFS blocks
DOCS_EXAMPLES = {
    "failing_block": {"summary": "Start of a block that later failed",
                      "value": {"events": ["E5", "E22"], "elapsed_seconds": 0.0}},
    "normal_looking_block": {"summary": "Typical normal write pipeline start",
                             "value": {"events": ["E5", "E22", "E5", "E5", "E11", "E9", "E11", "E9"], "elapsed_seconds": 1.0}},
    "invalid_event": {"summary": "Unknown event id (returns 422)",
                      "value": {"events": ["E99"], "elapsed_seconds": 1.0}},
}

app = FastAPI(title="HDFS early-warning", version=MODEL_VERSION)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(block: Annotated[BlockEvents, Body(openapi_examples=DOCS_EXAMPLES)]) -> Prediction:
    x = early_vector(block.events, block.elapsed_seconds)[None, :]
    p = float(predict_proba(x)[0])
    return Prediction(p_anomaly=p, alert=p >= THRESHOLD, threshold=THRESHOLD, model_version=MODEL_VERSION)
