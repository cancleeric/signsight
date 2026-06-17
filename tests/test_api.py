"""API tests using FastAPI's TestClient and an in-memory image.

These run against an untrained model (no checkpoint), so they assert the
contract -- status codes, JSON shape, error handling -- not accuracy.
"""

import io

from fastapi.testclient import TestClient
from PIL import Image

from serve.main import app

client = TestClient(app)


def _png_bytes(color=(200, 0, 0), size=(48, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["num_classes"] == 43


def test_predict_returns_topk():
    res = client.post(
        "/predict?top_k=5",
        files={"file": ("sign.png", _png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    preds = res.json()["predictions"]
    assert len(preds) == 5
    assert {"class_id", "label", "confidence"} <= preds[0].keys()
    assert all(0.0 <= p["confidence"] <= 1.0 for p in preds)


def test_predict_rejects_non_image():
    res = client.post(
        "/predict",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert res.status_code == 400


def test_index_served():
    assert client.get("/").status_code == 200
