"""Model: an EfficientNet backbone adapted for 7-class skin-lesion classification.

Transfer learning via `timm`: load EfficientNet pretrained on ImageNet, then replace
its 1000-class head with a fresh 7-class head. Passing `num_classes=7` to
`timm.create_model` performs that head swap automatically.

Run a sanity check:  uv run python src/model.py
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_model(
    num_classes: int = 7,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create an EfficientNet adapted to `num_classes`.

    freeze_backbone=True  -> "feature extraction": freeze the pretrained weights and
                             train only the new head. Fast; good when data is scarce.
    freeze_backbone=False -> "fine-tuning": update all weights. Usually higher accuracy
                             when you have enough data (our ~7k training images qualify).
    """
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    model = build_model()
    total, trainable = count_parameters(model)
    print("Backbone: efficientnet_b0 (ImageNet-pretrained, head -> 7 classes)")
    print(f"Parameters: total {total:,} | trainable {trainable:,}")
    print(f"Classifier head: {model.get_classifier()}")

    # Forward pass on a dummy batch to confirm output shape == (batch, num_classes).
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"\nForward pass: input {tuple(x.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (4, 7), "Unexpected output shape!"
    print("Output shape OK (one logit per class)")


if __name__ == "__main__":
    main()
