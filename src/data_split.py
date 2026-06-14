"""Create a reproducible, leakage-free train/val/test split for HAM10000.

Run: uv run python src/data_split.py

Design (see RESEARCH_LOG F1.1 & F1.2):
- Split by LESION, not by image, so no lesion's images land in two splits (no leakage).
- Stratify by diagnosis so every class keeps its proportion in train/val/test.

Trick that makes this simple: in HAM10000 all images of a lesion share the same
diagnosis, so a lesion is "atomic" with respect to the label. That lets us reduce a
(hard) grouped + stratified split over images to a (simple) stratified split over
lesions, then map the result back to images.

Output: data/splits.csv — the metadata plus a `split` column (train/val/test).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
META = DATA_DIR / "HAM10000_metadata.csv"
OUT = DATA_DIR / "splits.csv"

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def make_split() -> pd.DataFrame:
    df = pd.read_csv(META)

    # One row per lesion. Assert every lesion has a single diagnosis (the property
    # that lets us treat a lesion as one unit to stratify on).
    lesion_dx = df.groupby("lesion_id")["dx"].agg(["nunique", "first"])
    assert (lesion_dx["nunique"] == 1).all(), "A lesion maps to multiple diagnoses!"
    lesions = lesion_dx.reset_index()[["lesion_id", "first"]].rename(
        columns={"first": "dx"}
    )

    # Stratified split of LESIONS: carve out test first, then val from what remains.
    train_val, test = train_test_split(
        lesions, test_size=TEST_FRACTION, stratify=lesions["dx"], random_state=SEED
    )
    val_relative = VAL_FRACTION / (1.0 - TEST_FRACTION)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["dx"], random_state=SEED
    )

    # Map each lesion_id -> its split, then attach to every image row.
    split_of = {
        lid: name
        for name, part in [("train", train), ("val", val), ("test", test)]
        for lid in part["lesion_id"]
    }
    df = df.copy()
    df["split"] = df["lesion_id"].map(split_of)
    return df


def main() -> None:
    df = make_split()
    df.to_csv(OUT, index=False)

    # --- Verify the whole point: no lesion appears in more than one split ---
    crossing = df.groupby("lesion_id")["split"].nunique().gt(1).sum()
    assert crossing == 0, f"{crossing} lesions cross split boundaries!"
    print("Leakage check: 0 lesions cross split boundaries  OK\n")

    # --- Report split sizes and per-split class proportions ---
    print("Images per split:")
    print(df["split"].value_counts().to_string(), "\n")
    print("Class proportion within each split (%):")
    prop = (
        df.groupby("split")["dx"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .unstack()
        .fillna(0)
    )
    print(prop.to_string())
    print(f"\nSaved split assignments to {OUT}")


if __name__ == "__main__":
    main()
