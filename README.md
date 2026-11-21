# AI-Powered Intelligent Waste Classification and Segregation

A CPU-friendly college project that classifies the six original TrashNet categories: **cardboard, glass, metal, paper, plastic, and trash**. It provides a React dashboard, FastAPI prediction service, SQLite prediction history, reproducible dataset processing, model comparison, evaluation graphics, and optional Grad-CAM visual explanations.

`trash` means TrashNet's general-waste class. It is deliberately not renamed to organic waste.

## What is implemented

- React + Vite + Tailwind-based responsive dashboard with image preview, dashboard totals, history, class distribution, ranked predictions, low-confidence alert, disposal recommendation, and Grad-CAM display.
- FastAPI endpoints for health, model status, classification, history, analytics, and dashboard totals.
- SQLite storage at `backend/artifacts/waste_classifier.db` (created automatically).
- A custom CNN and MobileNetV2 transfer-learning implementation, early stopping, best-model checkpoints, saved metadata/class names, training curves, model metrics, per-class report, and confusion matrix.
- Dataset download, corrupt-image inspection, hash-based leakage check, and deterministic stratified 70/15/15 train/validation/test split. Image augmentation is present only in the Keras model's training path.
- Grad-CAM overlay generated from the currently loaded Keras model for each requested prediction.

## Repository layout

```text
backend/
  app/                 FastAPI, SQLite, inference, disposal advice, Grad-CAM
  ml/                  data download/inspection/preparation/training/evaluation
  artifacts/           generated models, reports, uploads, SQLite database
  tests/               API smoke tests
data/
  raw/                 downloaded TrashNet (not committed)
  processed/           generated split folders and manifest (not committed)
frontend/              Vite React dashboard
```

## Windows setup (PowerShell)

Prerequisites: Python 3.10 (the project was checked with 3.10.7) and Node.js 20+. This is CPU-only; CUDA is not required.

```powershell
cd "C:\Users\SAIRAM MALLIPUDI\Desktop\CSE_471_Project"
py -3.10 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
cd frontend
npm install
cd ..
```

If an interrupted package installation leaves `backend\.venv` unusable, close any Python processes and recreate it before retrying:

```powershell
Remove-Item -Recurse -Force backend\.venv
py -3.10 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

If PowerShell blocks virtual-environment activation, run this once for your user account and open a new terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Dataset preparation

From `backend\ml` with the virtual environment active:

```powershell
cd backend\ml
python download_trashnet.py
python inspect_dataset.py
python prepare_data.py
cd ..\..
```

The download script pulls the public TrashNet repository archive and locates its six named category folders. Before training, review `backend\artifacts\reports\dataset_inspection.json`; corrupted items are reported but never copied into the processed set. `prepare_data.py` fails on duplicate bytes, which prevents a duplicate image leaking into two splits. It produces `data\processed\manifest.csv` so every split is traceable.

Do **not** use `--force` unless you have reviewed the existing `data\processed` output; it replaces that generated split folder.

## CPU training and evaluation

Start with a short smoke run to confirm the pipeline. This creates genuine artifacts, but its results are not final model results:

```powershell
cd backend\ml
python train.py --model custom_cnn --epochs 1 --batch-size 8
```

Train both comparison models (12 epochs is a sensible initial CPU baseline; early stopping can finish earlier):

```powershell
python train.py --model custom_cnn --epochs 12 --batch-size 8
python train.py --model mobilenetv2 --epochs 12 --batch-size 8 --publish
python compare_models.py
cd ..\..
```

Publishing copies the chosen checkpoint to `backend\artifacts\models\best_model.keras` and writes matching metadata. You may instead publish the custom CNN by rerunning its command with `--publish`. The API does not make up a class or confidence when no published model exists—it responds with HTTP 503 and an exact training instruction.

Actual run outputs are written under `backend\artifacts\reports`:

- `<model>_metrics.json` — accuracy, weighted precision, recall, F1, and mean per-image inference time.
- `<model>_classification_report.csv` — per-class precision, recall, and F1.
- `<model>_confusion_matrix.png` and `<model>_training_graph.png`.
- `comparison.json` — both measured model results once both have run.

No dataset count, accuracy, confidence, or comparison result is committed or claimed before the scripts produce it.

## Run the application

Open two PowerShell windows in the project directory.

Terminal 1:

```powershell
backend\.venv\Scripts\Activate.ps1
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Terminal 2:

```powershell
cd frontend
npm run dev
```

Open the local URL shown by Vite, normally `http://localhost:5173`. Upload a JPEG, PNG, or WebP under 10 MB. The **Include Grad-CAM** control requests a per-image attention heatmap. Each successful classification is stored locally in SQLite.

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service and model-ready health check |
| GET | `/api/model-status` | Exact published-model status/classes |
| POST | `/api/classify?include_gradcam=true` | Multipart `image`; predicts and saves history |
| GET | `/api/predictions` | Recent prediction records |
| GET | `/api/analytics` | Totals, mean confidence, labels distribution |
| GET | `/api/dashboard` | Combined dashboard payload |

Example classification request after publishing a model:

```powershell
curl.exe -X POST "http://localhost:8000/api/classify?include_gradcam=false" -F "image=@C:\path\to\waste.jpg"
```

## Tests and verification

With the backend virtual environment active:

```powershell
cd backend
python -m pytest tests -q
python -m compileall app ml
cd ..\frontend
npm run build
```

The API tests cover health, empty history/analytics, and explicit model status. They are intentionally valid before model training. After training, use the curl request above to validate inference, SQLite history, and Grad-CAM on an actual image.

## Notes for extension

To add organic waste, acquire and document a distinct suitable dataset, include it in the split process, retrain a seven-class model, and update the class list. Do not relabel TrashNet's `trash` images as organic.
