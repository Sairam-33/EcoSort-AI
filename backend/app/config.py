from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ARTIFACTS_DIR / 'waste_classifier.db'}")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

for folder in (ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, UPLOADS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
