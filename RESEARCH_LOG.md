# Research Log — ISIC Skin Lesion Classifier

A running record of **observations → why they matter → decisions**, kept as we build
the project. This is a research journal, not documentation: it captures the *reasoning*
behind each design choice so the final model is defensible, reproducible, and
explainable — and so the thinking transfers to future projects.

**How to read each entry:**
- **Finding** — what we observed (with numbers; observations beat opinions)
- **Why it matters** — the consequence if ignored
- **Decision** — what we'll do about it, and when (which milestone)
- **Status** — `open` / `decided` / `implemented` / `verified`

---

## Milestone 1 — Data & EDA

### F1.1 — Severe class imbalance (58×)
- **Finding:** 7 classes, 10,015 images. `nv` (benign nevi) = 66.9%; `df` (dermatofibroma)
  = 1.1%. Largest/smallest ratio ≈ **58×**.
  ```
  nv 6705 (66.9%) | mel 1113 (11.1%) | bkl 1099 (11.0%) | bcc 514 (5.1%)
  akiec 327 (3.3%) | vasc 142 (1.4%) | df 115 (1.1%)
  ```
- **Why it matters:** A model predicting `nv` for everything scores ~67% accuracy while
  catching **zero melanomas**. Accuracy is therefore meaningless here; the minority
  classes (including malignant `mel`, `bcc`) are the clinically important ones.
- **Decision:**
  1. Use **AUC + per-class recall/precision**, not accuracy, as headline metrics. *(Milestone 5)*
  2. Handle imbalance in training via **class-weighted loss** (and possibly focal loss / resampling). *(Milestone 4)*
  3. Use a **stratified** split so every class is represented in train/val/test. *(Milestone 2)*
- **Status:** decided

### F1.2 — Multiple images per lesion → data-leakage risk
- **Finding:** 10,015 images but only **7,470 unique lesions**. 26.2% of lesions have
  >1 image (up to 6). Confirmed in metadata: rows share `lesion_id` with distinct `image_id`.
- **Why it matters:** A naive split *by image* can place different images of the **same
  physical lesion** in both train and test. The model then recognizes that specific lesion
  rather than generalizing, **inflating the test score** into a number we can't trust. This
  is a classic, credibility-destroying medical-imaging mistake.
- **Decision:** Split **by `lesion_id`, not by image** (grouped split) so all images of a
  lesion stay on one side. Combine with stratification (F1.1) →
  `StratifiedGroupKFold` / `GroupShuffleSplit`. *(Milestone 2)*
- **Status:** decided

### F1.3 — Images transfer well from natural-image pretraining
- **Finding:** Dermatoscopic RGB images, well-centered lesions, visually similar to
  natural photos. Some contain artifacts (hair, rulers, ink marks).
- **Why it matters:** ImageNet-pretrained backbones should transfer effectively (fast,
  data-efficient training). The artifacts are a *spurious-correlation* risk — a model could
  learn "ruler ⇒ malignant" instead of real lesion features.
- **Decision:**
  1. Use **transfer learning** from a pretrained backbone via `timm`. *(Milestone 3)*
  2. Check with **Grad-CAM** that the model attends to the lesion, not artifacts. *(Milestone 6)*
- **Status:** decided

---

## Milestone 2 — Data pipeline

### F2.1 — Leakage-free, stratified split implemented & verified
- **Finding/Decision:** Implemented the split from F1.1+F1.2. Key move: since every image
  of a lesion shares one diagnosis, a lesion is *atomic* w.r.t. the label — so we collapse
  images→lesions, do an ordinary **stratified** split on lesions, then map back to images.
  This reduces a hard grouped+stratified problem to a simple one.
- **Verification (seed 42):**
  - **0 lesions** cross split boundaries (asserted in code) → no leakage.
  - Sizes: train 7054 / val 1464 / test 1497 images (~70/15/15).
  - Class proportions ~identical across splits (e.g. `nv` ~67%, `mel` ~11%, `df` ~1.1–1.5%).
  - Image counts drift slightly from 70/15/15 because multi-image lesions cluster — expected.
- **Status:** verified  → output `data/splits.csv` (`split` column: train/val/test)

### F2.2 — Dataset & transforms (preprocessing decisions)
- **Decisions:**
  - **Fixed class ordering** (`akiec,bcc,bkl,df,mel,nv,vasc` → 0..6) so label indices never
    silently change between runs.
  - **ImageNet normalization** (mean/std) so inputs match the pretrained backbone's (M3)
    training distribution.
  - **Augmentation on train only** (RandomResizedCrop, H+V flips, ±20° rotation, mild color
    jitter). Vertical flip is valid because dermatoscopic lesions have no canonical orientation.
- **Verification:** train sample → tensor `(3,224,224)` float32, value range ≈ `[-1.9, 2.6]`
  (confirms normalization), label maps correctly. Split sizes match F2.1.
- **Status:** implemented  → `src/dataset.py`

---

## Milestone 3 — Model

### F3.1 — EfficientNet-B0 via transfer learning
- **Decision:** Use `timm` to load **EfficientNet-B0 pretrained on ImageNet**, swap its
  1000-class head for a fresh `Linear(1280 → 7)`. Default to **full fine-tuning** (all 4.0M
  params trainable), since ~7k training images is enough to adapt the backbone; expose a
  `freeze_backbone` flag to compare feature-extraction later.
- **Why:** Lesion images resemble natural images (F1.3), so ImageNet features transfer well.
  B0 is small/fast (good accuracy-per-compute) — a sensible first backbone before scaling up.
- **Verification:** forward pass `(4,3,224,224) → (4,7)`; total/trainable params = 4,016,515.
  Model outputs raw logits (softmax handled by the loss).
- **Status:** implemented  → `src/model.py`
- **To test later:** freeze vs fine-tune; B0 vs larger (B3) vs ResNet50. *(experiments)*

---

## Transferable principles (the "think like a researcher" list)

Generalizable lessons, accumulated as we go — these apply far beyond this project:

1. **Look at your data before you model.** EDA surfaced both the imbalance and the leakage
   trap *before* a single training run. Most serious bugs are data bugs.
2. **Pick metrics that match the real goal.** Imbalanced + asymmetric costs ⇒ accuracy lies.
   Choose metrics a domain expert would care about (catching melanoma > overall accuracy).
3. **Define "unseen" carefully.** Leakage hides in grouping structure (lesions, patients,
   hospitals). Ask "what unit must not cross the split boundary?" *before* splitting.
4. **Every design choice should trace to a finding.** When asked "why did you weight the
   loss?", the answer is F1.1, with numbers — not "it's common practice."

---

## Open questions / things to test later

- Binary (benign vs malignant) vs full 7-class — which framing best serves the goal? *(revisit M2/M5)*
- Does class weighting or focal loss work better for this imbalance? *(experiment in M4)*
- How much does input resolution (`--size`) matter vs compute cost? *(experiment later)*

---

*Updated through Milestone 1. Append new findings as each milestone produces them.*
