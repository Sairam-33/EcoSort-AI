import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from constants import CLASS_NAMES

def save_evaluation(model, dataset, model_name: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = [], []
    start = time.perf_counter()
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())
    seconds = time.perf_counter() - start
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {"model": model_name, "samples": len(y_true), "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))), "precision_weighted": float(precision), "recall_weighted": float(recall), "f1_weighted": float(f1), "average_inference_ms": (seconds / max(len(y_true), 1)) * 1000}
    (output_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T.to_csv(output_dir / f"{model_name}_classification_report.csv")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title=f"{model_name} confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]): axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image); figure.tight_layout(); figure.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=160); plt.close(figure)
    return metrics
