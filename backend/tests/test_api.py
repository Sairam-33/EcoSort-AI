"""Route-handler tests using the same SQLite session and upload objects as FastAPI."""
from io import BytesIO
import sys
from pathlib import Path
import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import SessionLocal
from app.main import analytics, classify, dashboard, health, model_status, predictions

def test_health_and_model_status():
    assert health()["status"] == "ok"
    assert isinstance(model_status()["ready"], bool)

def test_history_analytics_and_dashboard():
    with SessionLocal() as db:
        assert isinstance(predictions(limit=50, db=db), list)
        assert "total_predictions" in analytics(db=db)
        assert "model" in dashboard(db=db)

def test_classification_requires_a_published_model():
    upload = UploadFile(filename="sample.jpg", file=BytesIO(b"not-a-real-image"), headers=Headers({"content-type": "image/jpeg"}))
    with SessionLocal() as db:
        with pytest.raises(HTTPException) as response:
            classify(image=upload, include_gradcam=False, db=db)
    assert response.value.status_code == 503
    assert "Run the training command" in response.value.detail
