import json
import shutil
import uuid
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from .config import LOW_CONFIDENCE_THRESHOLD, MAX_UPLOAD_BYTES, REPORTS_DIR, UPLOADS_DIR
from .database import Base, engine, get_db
from .models import Prediction
from .schemas import ClassificationResponse, ModelStatus, PredictionHistoryItem, RankedPrediction
from .services.inference import ModelUnavailableError, classifier
from .services.recommendations import recommendation_for

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Intelligent Waste Classification API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://ecosort-pj1ieupl2-sairam-ce29.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok", "model": classifier.status()["ready"]}

@app.get("/api/model-status", response_model=ModelStatus)
def model_status():
    return classifier.status()

@app.post("/api/classify", response_model=ClassificationResponse)
def classify(image: UploadFile = File(...), include_gradcam: bool = Query(False), db: Session = Depends(get_db)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
    # A bounded direct read avoids an unnecessary thread-pool hop for small uploads.
    # It also works reliably with Windows' temporary upload files.
    content = image.file.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be between 1 byte and 10 MB.")
    try:
        result = classifier.predict(content, include_gradcam)
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    extension = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    stored_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{extension}"
    stored_path.write_bytes(content)
    record = Prediction(filename=Path(image.filename or "upload").name, image_path=str(stored_path), predicted_class=result["predicted_class"], confidence=result["confidence"], top_predictions_json=json.dumps(result["top_predictions"]), model_name=result["model_name"])
    db.add(record)
    db.commit()
    db.refresh(record)
    low = result["confidence"] < LOW_CONFIDENCE_THRESHOLD
    return ClassificationResponse(id=record.id, filename=record.filename, predicted_class=result["predicted_class"], confidence=result["confidence"], top_predictions=[RankedPrediction(**item) for item in result["top_predictions"]], recommendation=recommendation_for(result["predicted_class"]), low_confidence=low, warning="Low confidence: verify the item manually before disposal." if low else None, model_name=result["model_name"], gradcam_image=result.get("gradcam_image"))

@app.get("/api/predictions", response_model=list[PredictionHistoryItem])
def predictions(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return db.scalars(select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)).all()

@app.get("/api/analytics")
def analytics(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Prediction.id))) or 0
    average = db.scalar(select(func.avg(Prediction.confidence))) or 0.0
    rows = db.execute(select(Prediction.predicted_class, func.count(Prediction.id)).group_by(Prediction.predicted_class)).all()
    return {"total_predictions": total, "average_confidence": round(float(average), 4), "by_class": [{"label": label, "count": count} for label, count in rows]}

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)):
    result = analytics(db)
    result["model"] = classifier.status()
    result["report_available"] = (REPORTS_DIR / "comparison.json").exists()
    return result
