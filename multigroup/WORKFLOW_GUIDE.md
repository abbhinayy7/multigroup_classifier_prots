# MultiGroup ProteoBoostR - Complete Workflow Guide

## Overview
This guide explains how to run the complete multiclass proteomics classification pipeline using the multigroup CLI. The workflow includes:
1. **Data Generation** (optional): Create synthetic test data
2. **Model Training**: Bayesian optimization + XGBoost training
3. **Model Evaluation**: One-vs-Rest (OVR) and One-vs-One (OVO) ROC analysis
4. **Model Application**: Predict on independent datasets

---

## Quick Start

### Option A: PowerShell (Windows)
```powershell
cd f:\ProteoBoostR\multigroup
.\RUN_ALL_TASKS.ps1
```

### Option B: Bash/Shell (Linux/Mac/Git Bash)
```bash
cd /path/to/ProteoBoostR/multigroup
chmod +x run_all_tasks.sh
bash run_all_tasks.sh
```

---

## Detailed Command Explanations

### 1. TRAIN: Train Model with Hyperparameter Optimization

**Command:**
```bash
python py_scripts/cli.py train \
  --annotation <annotation_file.tsv> \
  --protein <protein_file.tsv> \
  --annotcol <label_column_name> \
  --output <output_directory> \
  --n_iter 12 \
  --init_points 3 \
  --testsize 0.25 \
  --seed 42
```

**Parameters Explained:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--annotation` | path | required | TSV file with sample IDs and labels (sample_id column + label column) |
| `--protein` | path | required | TSV file with protein intensities (rows=proteins, cols=sample IDs) |
| `--annotcol` | string | required | Name of the column in annotation file containing class labels |
| `--output` | path | required | Directory to save all outputs |
| `--n_iter` | int | 25 | Number of Bayesian optimization iterations (more = better params but slower) |
| `--init_points` | int | 5 | Number of random exploration points before Bayesian search |
| `--testsize` | float | 0.3 | Fraction of data reserved for testing (0-1) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--classes` | list | all | Optional: filter to specific classes (e.g., `--classes Control GroupA GroupB`) |

**What it does:**
- Reads annotation and protein data, merges on sample_id
- Splits into train/test sets (stratified by class)
- Optimizes 8 hyperparameters via Bayesian optimization:
  - `learning_rate`: 0.001-0.5
  - `max_depth`: 2-12
  - `subsample`: 0.4-1.0
  - `colsample_bytree`: 0.3-1.0
  - `min_child_weight`: 1-10
  - `gamma`: 0.0-5.0
  - `reg_alpha`: 0.0-10.0 (L1 regularization)
  - `reg_lambda`: 0.0-10.0 (L2 regularization)
- Trains final XGBoost model (1000 boosting rounds)
- Saves model, train/test matrices, and best parameters

**Example (Real Data):**
```bash
python py_scripts/cli.py train \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output ./gbm_output \
  --n_iter 25 --init_points 5 --testsize 0.25
```

---

### 2. EVALUATE: Assess Model Performance with ROC Analysis

**Command (One-vs-Rest):**
```bash
python py_scripts/cli.py evaluate \
  --model <path/to/xgb_model_<timestamp>.joblib> \
  --annotation <annotation_file.tsv> \
  --protein <protein_file.tsv> \
  --annotcol <label_column_name> \
  --output <output_directory> \
  --roc_mode ovr
```

**Command (One-vs-One Pairwise):**
```bash
python py_scripts/cli.py evaluate \
  --model <path/to/xgb_model_<timestamp>.joblib> \
  --annotation <annotation_file.tsv> \
  --protein <protein_file.tsv> \
  --annotcol <label_column_name> \
  --output <output_directory> \
  --roc_mode ovo
```

**Parameters Explained:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | path | required | Path to saved .joblib model file |
| `--annotation` | path | required | Annotation TSV (for true labels) |
| `--protein` | path | required | Protein data TSV |
| `--annotcol` | string | required | Label column name |
| `--output` | path | required | Output directory for results |
| `--roc_mode` | choice | ovr | `ovr` = One-vs-Rest, `ovo` = One-vs-One pairwise |
| `--classes` | list | all | Optional: filter to specific classes |

**What it does:**

**OVR Mode (Default):**
- Generates one-vs-rest ROC curves (one class vs all others)
- Computes macro-average AUC
- Single ROC PNG with all classes + macro line

**OVO Mode:**
- Generates pairwise ROC for each pair of classes
- Skips pairs with <10 samples
- Creates individual PNG files per pair (e.g., `roc_pair_ClassA_vs_ClassB_<timestamp>.png`)
- Saves summary TSV with pairwise AUCs

**Common Outputs (Both Modes):**
- `evaluation_report_<timestamp>.tsv` — per-class precision, recall, f1-score
- `confusion_matrix_<timestamp>.tsv` — classification matrix
- `predicted_probabilities_<timestamp>.tsv` — predictions ranked by confidence
- `predicted_samples_<timestamp>.png` — scatter plot of predictions by class

---

### 3. APPLY: Make Predictions on New Data

**Command:**
```bash
python py_scripts/cli.py apply \
  --model <path/to/xgb_model_<timestamp>.joblib> \
  --protein <new_protein_file.tsv> \
  --annotation <optional_annotation.tsv> \
  --annotcol <optional_label_column> \
  --output <output_directory>
```

**Parameters Explained:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | path | required | Path to saved .joblib model file |
| `--protein` | path | required | Protein data TSV (new samples) |
| `--annotation` | path | optional | Annotation TSV (for comparison/visualization) |
| `--annotcol` | string | optional | Label column (ignored if no annotation) |
| `--output` | path | required | Output directory |
| `--classes` | list | all | Optional: filter to specific classes |

**What it does:**
- Loads saved model and feature list
- Aligns input features to model's expected features
  - Missing features → filled with NaN
  - Extra features → ignored
- Generates predictions and probabilities
- Outputs:
  - `predicted_probabilities_<timestamp>_adhoc.tsv` — sample IDs, predictions, max probability
  - `predicted_samples_<timestamp>_adhoc.png` — ranked confidence plot

**Example (Cross-Dataset):**
```bash
python py_scripts/cli.py apply \
  --model ./gbm_output/xgb_model_20260130114430.joblib \
  --protein ../GBM_testcase/Werner_data.tsv \
  --annotation ../GBM_testcase/Werner_annot.tsv \
  --annotcol multiomic_splitted \
  --output ./gbm_output
```

---

## Output Files Explained

### Training Outputs
```
train_matrix_<timestamp>.tsv          # Training features + sample IDs (tsv)
test_matrix_<timestamp>.tsv           # Test features + sample IDs (tsv)
xgb_model_<timestamp>.joblib          # Serialized model (binary)
best_params_<timestamp>.tsv           # Optimized hyperparameters (tsv)
```

### Evaluation Outputs
```
evaluation_report_<timestamp>.tsv     # Per-class metrics (precision/recall/f1/support)
confusion_matrix_<timestamp>.tsv      # Classification confusion matrix
roc_curve_<timestamp>.png             # One-vs-Rest ROC curves (OVR mode)
roc_pair_A_vs_B_<timestamp>.png       # Pairwise ROC (OVO mode, multiple files)
pairwise_aucs_<timestamp>.tsv         # Pairwise AUC summary (OVO mode)
predicted_probabilities_<timestamp>.tsv # Ranked predictions + scores
predicted_samples_<timestamp>.png     # Ranked sample confidence scatter plot
```

### Apply Outputs
```
predicted_probabilities_<timestamp>_adhoc.tsv  # Predictions on new data
predicted_samples_<timestamp>_adhoc.png        # Confidence plot
```

---

## Data Format Requirements

### Annotation File (TSV)
- **Required columns:**
  - `sample_id` (first column or explicitly named)
  - Label column (any name, specified with `--annotcol`)
- **Example:**
  ```
  sample_id    multiomic_splitted
  C3L.00104    nmf1
  C3L.00365    nmf3
  C3L.00674    nmf1
  ```

### Protein File (TSV)
- **Format:** Rows = proteins, Columns = sample IDs
- **Index:** First column = protein names
- **Example:**
  ```
                  C3L.00104  C3L.00365  C3L.00674
  P04217          0.4        -0.55      3.29
  P01023          2.26       3.02       4.42
  Q9NRG9          -1.11      0.69       -0.46
  ```

---

## Typical Workflow Example

### Step 1: Train on Dataset A
```bash
python py_scripts/cli.py train \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output ./results \
  --n_iter 25 --init_points 5 --seed 42
```

### Step 2: Evaluate with Both ROC Modes
```bash
# OVR evaluation
python py_scripts/cli.py evaluate \
  --model ./results/xgb_model_20260130114430.joblib \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output ./results \
  --roc_mode ovr

# OVO evaluation
python py_scripts/cli.py evaluate \
  --model ./results/xgb_model_20260130114430.joblib \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output ./results \
  --roc_mode ovo
```

### Step 3: Apply to Dataset B (e.g., Werner)
```bash
python py_scripts/cli.py apply \
  --model ./results/xgb_model_20260130114430.joblib \
  --protein ../GBM_testcase/Werner_data.tsv \
  --annotation ../GBM_testcase/Werner_annot.tsv \
  --annotcol multiomic_splitted \
  --output ./results
```

---

## Tips & Best Practices

1. **Hyperparameter Tuning:**
   - Start with `--n_iter 10 --init_points 3` for quick tests
   - Increase to `--n_iter 25 --init_points 5` for production models
   - More iterations → better performance but longer computation

2. **Test Size:**
   - Use `--testsize 0.2-0.3` for small datasets (<100 samples)
   - Use `--testsize 0.2-0.25` for larger datasets
   - Stratified split ensures class balance in both sets

3. **ROC Mode Selection:**
   - **OVR:** Good for overall performance summary, single figure
   - **OVO:** Better for detailed pairwise comparisons, identify confusing class pairs

4. **Class Balance:**
   - If class imbalances exist, consider filtering with `--classes` to create balanced subsets
   - Warning issued if any class has <3 samples

5. **Feature Alignment:**
   - `apply` mode fills missing features with NaN (XGBoost can handle this)
   - For imputation, preprocess protein data before running `apply`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` | Check paths are absolute or relative correctly |
| `KeyError: <column_name>` | Verify column name matches exactly (case-sensitive) |
| `ValueError: Need at least two classes` | Ensure label column has ≥2 distinct values |
| Small class warning | Collect more samples or use filtering with `--classes` |
| OVO skips pairs | Pair has <10 samples; increase data or adjust threshold |

---

## Next Steps

- **Reproduce Results:** Run `RUN_ALL_TASKS.ps1` / `run_all_tasks.sh`
- **Custom Data:** Prepare annotation + protein TSV files following the format above
- **Visualization:** Open PNG files to inspect ROC curves and predictions
- **Metrics:** Review TSV reports for per-class and overall performance

