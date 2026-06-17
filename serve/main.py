"""FastAPI inference service for the signsight model.

    uvicorn serve.main:app --reload

Set SIGNSIGHT_CHECKPOINT to a trained checkpoint to serve real predictions;
without it the service loads a randomly-initialised model so the API and demo
page still run end to end (useful for tests and a quick smoke check).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from signsight.classes import CLASS_NAMES
from signsight.infer import load_model, predict
from signsight.model import build_model

# Use the bundled, ready-to-run model by default; override with the env var.
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "signsight-cnn.pt"
CHECKPOINT = os.getenv("SIGNSIGHT_CHECKPOINT") or (
    str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else None
)
DEVICE = "cpu"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="signsight", description="Traffic-sign recognition API", version="0.1.0")


def _load() -> torch.nn.Module:
    if CHECKPOINT and Path(CHECKPOINT).exists():
        return load_model(CHECKPOINT, DEVICE)
    # Fallback: untrained model so the service still boots and responds.
    return build_model("cnn", len(CLASS_NAMES)).to(DEVICE).eval()


model = _load()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "checkpoint_loaded": bool(CHECKPOINT and Path(CHECKPOINT).exists()),
        "num_classes": len(CLASS_NAMES),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), top_k: int = 3) -> dict:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="file is not a valid image")

    preds = predict(model, image, top_k=top_k, device=DEVICE)
    return {
        "filename": file.filename,
        "predictions": [
            {"class_id": p.class_id, "label": p.label, "confidence": round(p.confidence, 4)}
            for p in preds
        ],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Serve assets if any are added later.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
