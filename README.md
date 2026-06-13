# Skin Lesion Classifier (Dermatology / ISIC)

**Goal:** an end-to-end, shippable classifier that distinguishes skin lesion types
(e.g. benign vs malignant / melanoma) on real high-resolution dermatology images.

**Why this first:** ISIC images are RGB and behave like natural photos, so transfer
learning works beautifully — a confidence-building first "real" project that still
demands real medical-AI rigor (imbalance, calibration, explainability).

## Dataset

[ISIC Archive](https://www.isic-archive.com/) / ISIC Challenge datasets (skin lesion images
with diagnoses). Start with a manageable subset; scale up on a cloud GPU.

## Planned approach

- Pretrained backbone (ResNet50 / EfficientNet via `timm`), fine-tuned
- Class-imbalance handling: weighted or focal loss, balanced sampling
- Augmentation suited to dermoscopy (flips, rotations, color jitter)
- Train on a rented cloud GPU; prototype locally on `mps`

## Evaluation (the part that makes it portfolio-grade)

- ROC-AUC, sensitivity & specificity at a chosen operating point
- **Calibration** (reliability diagram) — are predicted probabilities trustworthy?
- Per-class performance + confusion matrix
- Grad-CAM heatmaps: is the model looking at the lesion, not artifacts?
- Honest error analysis: where and why it fails

## Deliverables

- [ ] `src/` training + eval pipeline (tested)
- [ ] Results table in this README
- [ ] Grad-CAM examples + demo GIF
- [ ] Gradio app (`app/`) + Dockerfile
- [ ] Short writeup / blog post

## Results

_(table goes here once trained)_
