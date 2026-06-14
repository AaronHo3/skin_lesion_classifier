"""PyTorch Dataset and image transforms for HAM10000.

A `Dataset` is the bridge between raw files and the model: given an index, it returns
one (image_tensor, label) pair. The DataLoader (built in the training script) batches
and shuffles these.

Run a sanity check:  uv run python src/dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS = DATA_DIR / "splits.csv"
IMG_DIRS = [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]

# Fixed, deterministic label ordering so class index 0..6 never changes between runs.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ImageNet statistics — the pretrained backbone (Milestone 3) was trained on images
# normalized with these exact values, so our inputs must match that distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, img_size: int = 224) -> transforms.Compose:
    """Build the preprocessing pipeline.

    Augmentation (random flips/rotation/color jitter) is applied to TRAINING images
    only — it multiplies the effective variety the model sees, reducing overfitting.
    Validation/test images are left unaltered so we measure performance on real inputs.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                # Dermatoscopic images have no canonical orientation, so BOTH flips
                # are valid label-preserving augmentations (unlike, say, photos of text).
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HAM10000Dataset(Dataset):
    """Serves (image_tensor, label_int) pairs for one split (train/val/test)."""

    def __init__(self, split: str, img_size: int = 224, augment: bool | None = None):
        if augment is None:
            augment = split == "train"  # augment training data only, by default

        self.df = (
            pd.read_csv(SPLITS).query("split == @split").reset_index(drop=True)
        )
        self.transform = build_transforms(train=augment, img_size=img_size)

        # Index every image file once so __getitem__ is a fast dict lookup.
        self._paths = {p.stem: p for folder in IMG_DIRS for p in folder.glob("*.jpg")}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(self._paths[row["image_id"]]).convert("RGB")
        x = self.transform(image)
        y = CLASS_TO_IDX[row["dx"]]
        return x, y


def main() -> None:
    """Sanity check: build each split and inspect one training sample."""
    for split in ("train", "val", "test"):
        ds = HAM10000Dataset(split)
        print(f"{split:5s}: {len(ds)} samples")

    ds = HAM10000Dataset("train")
    x, y = ds[0]
    print("\nOne training sample:")
    print(f"  image tensor: shape {tuple(x.shape)}, dtype {x.dtype}")
    print(f"  value range : [{x.min():.2f}, {x.max():.2f}]  (normalized, so ~[-2, 2])")
    print(f"  label       : {y} ({CLASSES[y]})")


if __name__ == "__main__":
    main()
