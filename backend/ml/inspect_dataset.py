import argparse
import json
from pathlib import Path
from PIL import Image
from constants import CLASS_NAMES

def inspect(root: Path) -> dict:
    report = {"dataset": str(root), "classes": {}, "corrupted_files": []}
    for label in CLASS_NAMES:
        folder = root / label
        images = sorted(path for path in folder.glob("*") if path.is_file()) if folder.is_dir() else []
        valid = 0
        for path in images:
            try:
                with Image.open(path) as image:
                    image.verify()
                valid += 1
            except Exception:
                report["corrupted_files"].append(str(path))
        report["classes"][label] = {"files": len(images), "valid": valid, "corrupted": len(images) - valid}
    report["total_valid"] = sum(item["valid"] for item in report["classes"].values())
    return report

def main():
    parser = argparse.ArgumentParser(description="Count TrashNet images and identify unreadable files.")
    parser.add_argument("--source", default="../../data/raw/trashnet")
    parser.add_argument("--output", default="../artifacts/reports/dataset_inspection.json")
    args = parser.parse_args()
    report = inspect(Path(args.source).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
