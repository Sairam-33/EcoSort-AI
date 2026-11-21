
"""Create deterministic, hash-checked 70/15/15 TrashNet splits."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from constants import CLASS_NAMES, SEED


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build reproducible TrashNet dataset splits."
    )

    parser.add_argument("--source", default="../../data/raw/trashnet")
    parser.add_argument("--output", default="../../data/processed")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(
            f"{output} is non-empty. Use --force only after reviewing it."
        )

    records = []

    for label in CLASS_NAMES:
        folder = source / label

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing category folder: {folder}")

        for path in sorted(folder.glob("*")):
            if path.is_file() and readable(path):
                records.append((path, label, sha256(path)))

    if not records:
        raise FileNotFoundError("No valid images found.")

    # Group identical image files by SHA-256.
    hash_groups = defaultdict(list)

    for record in records:
        hash_groups[record[2]].append(record)

    clean_records = []
    excluded = []

    for file_hash, group in hash_groups.items():

        categories = {record[1] for record in group}

        if len(categories) > 1:
            # Exclude all copies when identical images have conflicting labels.
            for record in group:
                excluded.append({
                    "file": str(record[0]),
                    "label": record[1],
                    "sha256": file_hash,
                    "reason": "Identical image assigned to different categories"
                })

        else:
            # Keep one copy if duplicates belong to the same category.
            clean_records.append(group[0])

    records = sorted(clean_records, key=lambda item: str(item[0]))

    if not records:
        raise ValueError("No images remain after duplicate checking.")

    print(f"Original valid images: {len(records) + len(excluded)}")
    print(f"Excluded ambiguous duplicate files: {len(excluded)}")
    print(f"Remaining unique images: {len(records)}")

    # Save duplicate audit report.
    report_path = output.parent / "processed_duplicate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(excluded, file, indent=2)

    # Stratified 70/15/15 split.
    indices = list(range(len(records)))
    labels = [record[1] for record in records]

    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=args.seed
    )

    holdout_labels = [labels[index] for index in holdout_idx]

    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.50,
        stratify=holdout_labels,
        random_state=args.seed
    )

    # Verify no duplicate hashes across splits.
    split_hashes = {
        split: {records[i][2] for i in split_indices}
        for split, split_indices in {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        }.items()
    }

    assert split_hashes["train"].isdisjoint(split_hashes["val"])
    assert split_hashes["train"].isdisjoint(split_hashes["test"])
    assert split_hashes["val"].isdisjoint(split_hashes["test"])

    # Replace existing processed data only after validation.
    if output.exists():
        shutil.rmtree(output)

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    rows = []

    for split, split_indices in splits.items():

        for label in CLASS_NAMES:
            (output / split / label).mkdir(
                parents=True,
                exist_ok=True
            )

        for index in split_indices:
            source_path, label, file_hash = records[index]

            destination = (
                output / split / label /
                f"{index:05d}_{source_path.name}"
            )

            shutil.copy2(source_path, destination)

            rows.append({
                "split": split,
                "label": label,
                "source": str(source_path),
                "file": str(destination),
                "sha256": file_hash
            })

    # Save complete dataset manifest.
    manifest_path = output / "manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["split", "label", "source", "file", "sha256"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSplit results:")

    for split, split_indices in splits.items():
        print(f"{split}: {len(split_indices)} images")

    print("\nImages per category:")

    for label in CLASS_NAMES:
        counts = {
            split: sum(
                1 for i in split_indices
                if records[i][1] == label
            )
            for split, split_indices in splits.items()
        }

        print(f"{label}: {counts}")

    print("\nDuplicate hash verification: PASSED")
    print(f"Manifest: {manifest_path}")
    print(f"Duplicate report: {report_path}")
    print(f"\nDataset successfully prepared at: {output}")


if __name__ == "__main__":
    main()
