"""Download the HAM10000 skin-lesion dataset from Kaggle into ./data.

Run once:  uv run python src/download_data.py

Requires Kaggle API credentials at ~/.kaggle/kaggle.json. The dataset is ~5.6 GB
(10,015 dermatoscopic images + a metadata CSV), so the first run takes a while.
The download is skipped automatically if the metadata file is already present.
"""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "kmader/skin-cancer-mnist-ham10000"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
METADATA = DATA_DIR / "HAM10000_metadata.csv"


def main() -> None:
    if METADATA.exists():
        print(f"Data already present at {DATA_DIR} — skipping download.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json

    print(f"Downloading {DATASET} -> {DATA_DIR}  (~5.6 GB, please be patient)...")
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True)
    print("Done.")


if __name__ == "__main__":
    main()
