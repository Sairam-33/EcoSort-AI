from datetime import datetime
from pydantic import BaseModel, Field

class RankedPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)

class ClassificationResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    top_predictions: list[RankedPrediction]
    recommendation: str
    low_confidence: bool
    warning: str | None = None
    model_name: str
    gradcam_image: str | None = None

class PredictionHistoryItem(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    model_name: str
    created_at: datetime

class ModelStatus(BaseModel):
    ready: bool
    message: str
    model_name: str | None = None
    classes: list[str] = []
