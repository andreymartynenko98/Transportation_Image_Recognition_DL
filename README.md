# Deep Learning for Transportation Image Recognition: A Custom CNN vs. VGG16 Approach

This repository contains **Project Two** for my portfolio: an image-classification study comparing a **custom CNN** trained from scratch against **VGG16** (transfer learning) on a small transportation dataset (airplane vs. car, plus other classes if present). The goal is to measure the trade‑offs between capacity, training time, and generalization.

> Notebook file: `ProjectTwo_DL.ipynb`

---

## 🗂️ Dataset

- **Source (Kaggle):** https://www.kaggle.com/datasets/abtabm/multiclassimagedatasetairplanecar  
- **Task:** Multiclass image recognition for transportation classes (e.g., airplane, car, etc.).
- **Expected folder structure (after you download/extract into `data/`):**
  ```
  data/
    train/
      airplane/
      car/
      ...
    val/            # optional; if absent, notebook will split train into train/val
      airplane/
      car/
      ...
    test/           # optional
      airplane/
      car/
      ...
  ```

---

## 🧠 Approach

- **Model A — Custom CNN:** A lightweight convolutional network trained from scratch with data augmentation.
- **Model B — VGG16 (transfer learning):** Pretrained on ImageNet, with a custom classification head; compare **feature extractor (frozen)** vs **fine‑tuning**.
- **Preprocessing & Augmentation:** Resize, center/standardize, random flips/rotations (and optionally color jitter).
- **Evaluation Metrics:** Accuracy, macro‑averaged Precision/Recall/F1; confusion matrix. If class imbalance exists, report per‑class metrics.
- **Reporting:** Compare validation metrics, training curves, and parameter counts; discuss when to prefer transfer learning vs. scratch models.

---

## 🔧 Setup

### Option A — `pip`

```bash
# from repo root
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

jupyter notebook
# Open ProjectTwo_DL.ipynb
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate project-two-dl
jupyter notebook
```

> GPU acceleration is recommended but not required. For Apple Silicon, use `tensorflow-macos` and `tensorflow-metal` (see comments in `requirements.txt`). For NVIDIA GPUs, ensure CUDA/cuDNN versions match your TensorFlow install.

---

## ▶️ How to Run

1. **Download** the Kaggle dataset and extract it under `data/` as shown above.
2. Activate your environment (pip or conda).
3. Launch Jupyter and open `ProjectTwo_DL.ipynb`.
4. Run cells to:
   - Build and train the **Custom CNN**.
   - Load **VGG16** (pretrained), attach a custom head, and train (frozen → fine‑tune).
   - Evaluate and compare metrics; save plots to `figures/` and optional weights to `checkpoints/`.

---

## 📂 Repository Structure

```
.
├── ProjectTwo_DL.ipynb      # Main DL notebook (provided)
├── data/                    # Place Kaggle dataset here (contains .gitkeep)
├── figures/                 # Training curves, confusion matrices
├── checkpoints/             # Optional model weights
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment (optional)
├── .gitignore               # Ignore caches, venvs, data dumps, weights
├── LICENSE                  # MIT
└── README.md                # This file
```

---

## ✅ Checklist

- [ ] Dataset placed under `data/` with class subfolders.
- [ ] Custom CNN and VGG16 runs complete; metrics recorded.
- [ ] Curves and confusion matrices saved to `figures/`.
- [ ] Results and observations summarized in README or notebook.
- [ ] Environment files updated if additional packages are used.

---

## 🔒 License

Released under the **MIT License** (see `LICENSE`).

© 2025 Andrey Martynenko
