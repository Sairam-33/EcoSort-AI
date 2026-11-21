
"""Download and organize the official resized TrashNet dataset."""

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from constants import CLASS_NAMES

DEFAULT_URL = (
    "https://github.com/garythung/trashnet/raw/master/"
    "data/dataset-resized.zip"
)


def locate_dataset(root: Path) -> Path:
    for path in root.rglob("*"):
        if path.is_dir() and all(
            (path / label).is_dir() for label in CLASS_NAMES
        ):
            return path

    raise FileNotFoundError(
        "Could not find all six TrashNet class folders in the archive."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize TrashNet."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        default="../../data/raw/trashnet"
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty folder: {output}"
        )

    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "trashnet.zip"

        print("Downloading the official TrashNet dataset...")
        print(f"URL: {args.url}")

        request = urllib.request.Request(
            args.url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            with open(archive, "wb") as file:
                shutil.copyfileobj(response, file)

        print(f"Downloaded size: {archive.stat().st_size / (1024 * 1024):.2f} MB")

        if not zipfile.is_zipfile(archive):
            raise RuntimeError(
                "Downloaded file is not a valid ZIP archive."
            )

        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(temp)

        source = locate_dataset(Path(temp))

        output.mkdir(parents=True, exist_ok=True)

        for label in CLASS_NAMES:
            shutil.copytree(
                source / label,
                output / label
            )

            count = sum(
                1 for file in (output / label).rglob("*")
                if file.is_file()
            )

            print(f"{label}: {count} files")

    print("\nTrashNet dataset downloaded successfully!")
    print(f"Dataset location: {output}")


if __name__ == "__main__":
    main()
