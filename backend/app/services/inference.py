import json
import threading
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from ..config import MODELS_DIR
from .gradcam import make_gradcam_data_url

class ModelUnavailableError(RuntimeError):
    pass

class WasteClassifier:
    """Lazy model loader so API health/history work before training."""
    def __init__(self):
        self._model = None
        self._metadata = None
        self._lock = threading.Lock()

    @property
    def metadata_path(self) -> Path:
        return MODELS_DIR / "best_model_metadata.json"

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.metadata_path.exists():
                raise ModelUnavailableError("No trained model is published. Run the training command with --publish first.")
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            model_path = MODELS_DIR / metadata["model_file"]
            if not model_path.exists():
                raise ModelUnavailableError(f"Model file is missing: {model_path.name}")
            import tensorflow as tf
            self._model = tf.keras.models.load_model(model_path, compile=False)
            self._metadata = metadata

    def status(self) -> dict:
        try:
            self._load()
            return {"ready": True, "message": "Trained model loaded.", "model_name": self._metadata["architecture"], "classes": self._metadata["class_names"]}
        except ModelUnavailableError as exc:
            return {"ready": False, "message": str(exc), "model_name": None, "classes": []}

    def predict(self, content: bytes, include_gradcam: bool = False) -> dict:
        self._load()
        try:
            image = ImageOps.exif_transpose(Image.open(BytesIO(content))).convert("RGB")
        except Exception as exc:
            raise ValueError("The uploaded file is not a readable image.") from exc
        original = np.asarray(image)
        size = tuple(self._metadata.get("image_size", [224, 224]))
        resized = image.resize(size)
        batch = np.asarray(resized, dtype=np.float32)[None, ...]
        if self._metadata.get("preprocessing") == "mobilenet_v2":
            import tensorflow as tf
            batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        else:
            batch /= 255.0
        scores = self._model.predict(batch, verbose=0)[0]
        indices = np.argsort(scores)[::-1][:3]
        labels = self._metadata["class_names"]
        ranked = [{"label": labels[int(index)], "confidence": float(scores[index])} for index in indices]
        result = {"predicted_class": ranked[0]["label"], "confidence": ranked[0]["confidence"], "top_predictions": ranked, "model_name": self._metadata["architecture"]}
        if include_gradcam:
            result["gradcam_image"] = make_gradcam_data_url(self._model, batch, original, int(indices[0]))
        return result

classifier = WasteClassifier()
