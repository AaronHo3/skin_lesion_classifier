"""Train the EfficientNet skin-lesion classifier.

Reuses the Stage 0 training pattern (seeding, best-checkpoint, train/eval loop) and
adds what HAM10000 demands: DataLoaders over our HAM10000Dataset, a CLASS-WEIGHTED loss
to counter the 58x imbalance (RESEARCH_LOG F1.1), and macro-AUC as the checkpoint metric
(accuracy is misleading on imbalanced data).

Local smoke test:  uv run python src/train.py --epochs 1
Full run (GPU)  :  uv run python src/train.py --epochs 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLASSES, HAM10000Dataset
from model import build_model

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def class_weights(train_ds: HAM10000Dataset, device: str) -> torch.Tensor:
    """Inverse-frequency weights so rare classes (mel, df, ...) count more in the loss.

    sklearn's 'balanced' scheme sets weight[c] = n_samples / (n_classes * count[c]),
    so a class with 1/58 the samples gets ~58x the weight. This stops the model from
    minimizing loss by simply always predicting the majority class `nv`.
    """
    labels = train_ds.df["dx"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(split: str, batch_size: int, workers: int, img_size: int) -> DataLoader:
    ds = HAM10000Dataset(split, img_size=img_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float, float]:
    """Return (loss, accuracy, macro-AUC) over a split."""
    model.eval()
    running = 0.0
    correct = total = 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running += criterion(logits, labels).item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        correct += (probs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    # One-vs-rest macro AUC: average the per-class AUC, so every class counts equally
    # regardless of size — the right summary metric for imbalanced data.
    macro_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return running / total, correct / total, macro_auc


def plot_history(history: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)
    ax_auc.plot(epochs, history["val_auc"], marker="o", color="green")
    ax_auc.set(xlabel="epoch", ylabel="macro-AUC", title="Validation macro-AUC")
    ax_auc.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HAM10000 classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | seed: {args.seed} | backbone: {args.backbone}")

    train_loader = make_loader("train", args.batch_size, args.workers, args.img_size)
    val_loader = make_loader("val", args.batch_size, args.workers, args.img_size)
    test_loader = make_loader("test", args.batch_size, args.workers, args.img_size)

    model = build_model(
        num_classes=len(CLASSES),
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    weights = class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = ROOT / args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.backbone}_best.pt"

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch:2d}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_auc {val_auc:.4f}"
            f"{'  <- best (saved)' if improved else ''}"
        )
        for key, value in zip(history, (epoch, train_loss, val_loss, val_auc)):
            history[key].append(value)

    plot_history(history, ROOT / "plots" / "training_curves.png")

    # Final test on the best checkpoint (not the last epoch).
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"\nBest epoch {best_epoch} (val macro-AUC {best_auc:.4f})")
    print("=== Test set ===")
    print(f"loss {test_loss:.4f} | accuracy {test_acc:.3f} | macro-AUC {test_auc:.4f}")


if __name__ == "__main__":
    main()
