# Proteomics Classification Toolkit 

Welcome! This repository contains a complete Python reimplementation of the ProteoBoostR workflow (originally a Shiny app) for training, evaluating, and applying XGBoost classifiers to proteomics datasets. The Python CLI provides reproducible training (Bayesian hyperparameter optimization), thorough evaluation (ROC, AUC, confusion matrices), and application of models to new datasets with clear, publication-quality visualizations.

---

## 🚀 Highlights

- Full reproducible pipeline: data merging → preprocessing → training → evaluation → application
- XGBoost classifier with Bayesian hyperparameter optimization
- **Enhanced visualizations**: publication-ready ROC and ranked-sample plots (150 DPI, labeled operating point, metrics box)
- Robust, user-friendly CLI: `train`, `evaluate`, `apply` subcommands
- Per-run logging, TSV outputs for all matrices and metrics, PNG visualizations

---

## 🧭 Quick Start (3 steps)

1. Inspect sample test data in `GBM_testcase/` (e.g., `Werner_annot.tsv`, `Werner_data.tsv`).
2. Train model:

```bash
python py_scripts/cli.py train \
  --annotation GBM_testcase/Werner_annot.tsv \
  --protein GBM_testcase/Werner_data.tsv \
  --annotcol subtypes --neg subtype1 --pos "subtype2+3+4" \
  --output GBM_testcase/improved_model --n_iter 15 --init_points 8
```

3. Evaluate or apply model:

```bash
python py_scripts/cli.py evaluate --model GBM_testcase/improved_model/xgb_model_*.json \
  --annotation GBM_testcase/Werner_annot.tsv --protein GBM_testcase/Werner_data.tsv \
  --annotcol subtypes --neg subtype1 --pos "subtype2+3+4" --output GBM_testcase/improved_model

python py_scripts/cli.py apply --model GBM_testcase/improved_model/xgb_model_*.json \
  --protein GBM_testcase/CPTAC_data.tsv --annotation GBM_testcase/CPTAC_annot.tsv \
  --annotcol multiomic_splitted --neg nmf1 --pos "nmf2+3+4" --evaltsv GBM_testcase/improved_model/evaluation_results_*.tsv \
  --band 0.1 --output GBM_testcase/improved_model
```

---

## 📁 What the CLI Produces

- `xgb_model_<ts>.json` — trained model (JSON)
- `best_params_<ts>.tsv` — best hyperparameters from Bayesian optimization
- `train_matrix_<ts>.tsv`, `test_matrix_<ts>.tsv` — data used for training/evaluation
- `predicted_probabilities_<ts>.tsv` — predicted probabilities per sample
- `evaluation_results_<ts>.tsv` — accuracy, sensitivity, specificity, AUC, best threshold
- `confusion_matrix_<ts>.tsv` — confusion matrix saved as TSV
- `predicted_samples_<ts>.png` — ranked-sample plot (enhanced)
- `roc_curve_<ts>.png` — ROC curve (enhanced)
- `proteoboostr_<ts>.log` — per-run log file

All outputs are saved into the directory you pass as `--output`.

---

## 🧩 Project Overview

This repository contains two integrated implementations:

1. The original **R Shiny** app (kept for reference). It accepts uploads via UI and performs the full workflow in-browser.
2. A **Python CLI** (in `py_scripts/`) re-implements the workflow for automation and reproducibility with these components:
   - `utils.py` – data loading, merging, preprocessing, and helper utilities
   - `cli.py` – unified CLI with `train`, `evaluate`, and `apply` subcommands
   - Helper scripts and documentation for reproducibility and testing

Why the Python version?
- Better automation for batch runs and deployment
- Easier to integrate into CI/CD and pipelines
- Improved visualizations and hyperparameter control

---

## � Docker Setup (Recommended for Reproducibility)

For fully reproducible runs across environments, use Docker:

```bash
# Build the image
docker build -f Dockerfile.multigroup -t multigroup_classifier:latest .

# Run the default test (compares binary vs multigroup classification)
docker run --rm -v "$(pwd)/output:/app/output" \
  --memory=8g --cpus=4 multigroup_classifier:latest

# Open an interactive shell to explore
docker run --rm -it -v "$(pwd):/app" multigroup_classifier:latest /bin/bash
```

**Benefits:**
- Same Python 3.11 + dependencies everywhere (Windows, macOS, Linux)
- No local dependency conflicts
- Isolated environment for testing

For full Docker instructions, resource limits, volume mounting, and troubleshooting, see **[README_DOCKER.md](README_DOCKER.md)**.

---

## �🛠️ How Training Works (short)

- Data merge: annotation (samples) + protein matrix (features)
- Preprocessing: filter classes, convert to numeric, remove NA-heavy features
- Train/Test split: stratified split by class
- Bayesian optimization: maximize CV AUC across parameter bounds
- Final training: train on the training split with best params
- Evaluation: compute AUC, Youden threshold, confusion matrix, and save plots

For details, see `py_scripts/TRAINING_REPORT.md`.

---

## 🎨 Visualization Notes

- ROC curves include an **operating point** (optimal threshold) and a **metrics box** showing Accuracy, Sensitivity, Specificity, Precision, AUC
- Ranked-sample plots show sample ranks by probability with **threshold shading** (positive/negative regions) and legend with counts
- All PNG outputs are saved at **150 DPI** and are suitable for publication

---

### Example Visualizations

**Enhanced ROC Curve**

![ROC Curve — Enhanced](/GBM_testcase/improved_model/roc_curve_20260128165741.png)
*Figure: Enhanced ROC curve showing AUC, operating point (threshold), and a metrics box with Accuracy/Sensitivity/Specificity/Precision.*

**Ranked Predicted Samples**

![Ranked Predicted Samples — Enhanced](/GBM_testcase/improved_model/predicted_samples_20260128165741.png)
*Figure: Samples ranked by predicted probability with the decision threshold and positive/negative confidence regions shaded.*

---

---

## 🎯 Tips for Improving Performance

- Increase `--n_iter` and `--init_points` during `train` to explore more hyperparameter space
- Provide more samples if possible (external datasets for validation are recommended)
- Use `apply` on independent cohorts to verify generalization
- Examine `train_matrix_*.tsv` and `test_matrix_*.tsv` to spot data or class imbalance

---

## 🧪 Quick Troubleshooting

- If a CLI command fails: check the per-run log `proteoboostr_<ts>.log` in the output folder
- Missing columns: ensure `annotation` file first column is `sample_id` and `annotcol` exists
- Feature mismatch during `apply`: verify protein IDs and sample IDs align between training and new data

---

## 📚 Where to look next

- `GBM_testcase/` — example datasets and outputs
- `py_scripts/cli.py` — CLI implementation (entry point)
- `py_scripts/utils.py` — processing helpers
- `GBM_testcase/improved_model/` — example of a finalized, high-quality model and visuals
- `IMPROVED_MODEL_REPORT.md` & `COMPLETE_STATUS_REPORT.md` — detailed analysis and results

---


