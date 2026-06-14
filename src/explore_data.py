"""Exploratory data analysis (EDA) for the HAM10000 dataset.

Run: uv run python src/explore_data.py

Prints the class distribution and lesion-level statistics to the terminal, and
saves two figures to plots/: the class distribution and one sample image per class.
EDA is the step where you *look* at your data before modeling t surfaces problems
(here: severe class imbalance and duplicate images per lesion) that dictate every
later design decision.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to file, no GUI

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
IMG_DIRS = [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]

# Human-readable names for the 7 diagnosis codes in the `dx` column.
CLASS_NAMES = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}


def image_path(image_id: str) -> Path:
    """Find an image file by id across the two HAM10000 image folders."""
    for folder in IMG_DIRS:
        candidate = folder / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image file for {image_id}")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR / "HAM10000_metadata.csv")

    print(f"Total images  : {len(df)}")
    print(f"Unique lesions: {df['lesion_id'].nunique()}")
    print(f"Columns       : {list(df.columns)}\n")

    # --- Class distribution: how imbalanced is the dataset? ---
    counts = df["dx"].value_counts()
    print("Class distribution:")
    for code, n in counts.items():
        name = CLASS_NAMES.get(code, code)
        print(f"  {code:6s} {name:22s} {n:5d}  ({n / len(df) * 100:5.1f}%)")
    print(f"  imbalance ratio (largest / smallest): {counts.max() / counts.min():.1f}x")

    # --- Lesion-level duplication: the data-leakage risk for splitting ---
    per_lesion = df.groupby("lesion_id").size()
    multi = (per_lesion > 1).sum()
    print(f"\nImages per lesion: mean {per_lesion.mean():.2f}, max {per_lesion.max()}")
    print(f"Lesions with >1 image: {multi} ({multi / len(per_lesion) * 100:.1f}% of lesions)")

    # --- Figure 1: class distribution bar chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color="steelblue")
    ax.set_title("HAM10000 class distribution")
    ax.set_xlabel("diagnosis (dx)")
    ax.set_ylabel("image count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)

    # --- Figure 2: one example image per class ---
    fig, axes = plt.subplots(1, len(counts), figsize=(2.2 * len(counts), 2.6))
    for ax, code in zip(axes, counts.index):
        sample_id = df.loc[df["dx"] == code, "image_id"].iloc[0]
        ax.imshow(Image.open(image_path(sample_id)))
        ax.set_title(code)
        ax.axis("off")
    fig.suptitle("One sample image per class")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sample_per_class.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved figures to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
