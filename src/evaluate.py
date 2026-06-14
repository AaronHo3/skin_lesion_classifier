"""Clinical evaluation of the trained classifier on the held-out test set.

Loads the best checkpoint and reports the metrics that matter for a medical model:
per-class precision/recall/F1, per-class AUC, a confusion matrix, melanoma recall, and
a malignant-vs-benign sensitivity/specificity. The headline macro-AUC (from training)
tells us ranking quality; THIS tells us whether the model is clinically safe.

Run: uv run python src/evaluate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import CLASSES, HAM10000Dataset
from model import build_model

ROOT = Path(__file__).resolve().parents[1]

# Clinically malignant / pre-malignant classes (the ones we must not miss).
MALIGNANT = {"mel", "bcc", "akiec"}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; return (probs [N,7], true_labels [N])."""
    model.eval()
    probs_list, labels_list = [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        probs_list.append(probs.cpu())
        labels_list.append(labels)
    return torch.cat(probs_list).numpy(), torch.cat(labels_list).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / f"{args.backbone}_best.pt"

    test_loader = DataLoader(
        HAM10000Dataset("test"), batch_size=args.batch_size, num_workers=2
    )
    # pretrained=False: we immediately overwrite weights with our checkpoint anyway.
    model = build_model(len(CLASSES), backbone=args.backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    probs, labels = collect_predictions(model, test_loader, device)
    preds = probs.argmax(axis=1)

    # --- Per-class precision / recall / F1 ---
    print("Per-class report (test set):")
    print(classification_report(labels, preds, target_names=CLASSES, digits=3))

    # --- Per-class one-vs-rest AUC ---
    print("Per-class AUC (one-vs-rest):")
    for i, name in enumerate(CLASSES):
        auc = roc_auc_score((labels == i).astype(int), probs[:, i])
        print(f"  {name:6s} {auc:.3f}")

    # --- Melanoma recall: the single most important clinical number ---
    mel = CLASSES.index("mel")
    mel_recall = ((preds == mel) & (labels == mel)).sum() / (labels == mel).sum()
    print(f"\nMelanoma recall (sensitivity): {mel_recall:.3f}")

    # --- Collapse to malignant vs benign and report sensitivity/specificity ---
    mal_idx = [CLASSES.index(c) for c in MALIGNANT]
    true_mal = np.isin(labels, mal_idx)
    pred_mal = np.isin(preds, mal_idx)
    tp = int((pred_mal & true_mal).sum())
    fn = int((~pred_mal & true_mal).sum())
    tn = int((~pred_mal & ~true_mal).sum())
    fp = int((pred_mal & ~true_mal).sum())
    print(
        f"Malignant-vs-benign: sensitivity {tp / (tp + fn):.3f} | "
        f"specificity {tn / (tn + fp):.3f}  (FN={fn} missed malignant)"
    )

    # --- Row-normalized confusion matrix (diagonal = per-class recall) ---
    cm = confusion_matrix(labels, preds, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f", xticks_rotation=45)
    ax.set_title("Confusion matrix (row-normalized = recall)")
    fig.tight_layout()
    out = ROOT / "plots" / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved confusion matrix to {out}")


if __name__ == "__main__":
    main()
