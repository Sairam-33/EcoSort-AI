import json
from pathlib import Path

def main():
    reports = Path(__file__).resolve().parents[1] / "artifacts" / "reports"
    results = []
    for name in ("custom_cnn", "mobilenetv2"):
        path = reports / f"{name}_metrics.json"
        if path.exists(): results.append(json.loads(path.read_text(encoding="utf-8")))
    if len(results) < 2:
        raise FileNotFoundError("Train and evaluate both models before comparing them.")
    (reports / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
