# MultiGroup ProteoBoostR - Quick Command Reference

## 🚀 Fastest Way to Get Started

### Windows (PowerShell)
```powershell
cd f:\ProteoBoostR\multigroup
.\RUN_ALL_TASKS.ps1
```

### Linux/Mac (Bash)
```bash
cd /path/to/ProteoBoostR/multigroup
bash run_all_tasks.sh
```

This runs the complete pipeline:
1. Generates synthetic test data
2. Trains multiclass model
3. Evaluates with OVR ROC
4. Evaluates with OVO pairwise ROC
5. Makes predictions on independent data

---

## 📋 Individual Commands (Copy & Paste)

### 1️⃣ TRAIN - Build Multiclass Model

**Synthetic Data:**
```bash
python py_scripts/cli.py train \
  --annotation test_data/annotation.tsv \
  --protein test_data/protein.tsv \
  --annotcol label \
  --output test_output \
  --n_iter 3 --init_points 2 --seed 42
```

**Real Data (GBM CPTAC):**
```bash
python py_scripts/cli.py train \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output gbm_output \
  --n_iter 12 --init_points 3 --seed 42
```

**Custom Data:**
```bash
python py_scripts/cli.py train \
  --annotation YOUR_ANNOT.tsv \
  --protein YOUR_PROTEIN.tsv \
  --annotcol LABEL_COLUMN_NAME \
  --output output_dir \
  --n_iter 25 --init_points 5 --testsize 0.25 --seed 42
```

---

### 2️⃣ EVALUATE - One-vs-Rest ROC (OVR)

```bash
python py_scripts/cli.py evaluate \
  --model output_dir/xgb_model_20260130114430.joblib \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output output_dir \
  --roc_mode ovr
```

---

### 3️⃣ EVALUATE - One-vs-One Pairwise ROC (OVO)

```bash
python py_scripts/cli.py evaluate \
  --model output_dir/xgb_model_20260130114430.joblib \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output output_dir \
  --roc_mode ovo
```

---

### 4️⃣ APPLY - Predict on New Dataset

```bash
python py_scripts/cli.py apply \
  --model output_dir/xgb_model_20260130114430.joblib \
  --protein ../GBM_testcase/Werner_data.tsv \
  --annotation ../GBM_testcase/Werner_annot.tsv \
  --annotcol multiomic_splitted \
  --output output_dir
```

---

## 🎯 Command Parameter Guide

### TRAIN Parameters
```
--annotation         PATH    Required   Annotation TSV (sample_id + labels)
--protein            PATH    Required   Protein TSV (rows=proteins, cols=samples)
--annotcol           STRING  Required   Label column name
--output             PATH    Required   Output directory
--n_iter             INT     25         BO iterations (higher = better, slower)
--init_points        INT     5          Random init points before BO
--testsize           FLOAT   0.3        Test fraction (0-1)
--seed               INT     42         Random seed
--classes            LIST    all        Filter to classes: --classes Class1 Class2
```

### EVALUATE Parameters
```
--model              PATH    Required   Model .joblib file
--annotation         PATH    Required   Annotation TSV
--protein            PATH    Required   Protein TSV
--annotcol           STRING  Required   Label column name
--output             PATH    Required   Output directory
--roc_mode           CHOICE  ovr        'ovr' (One-vs-Rest) or 'ovo' (One-vs-One)
--classes            LIST    all        Filter to classes
```

### APPLY Parameters
```
--model              PATH    Required   Model .joblib file
--protein            PATH    Required   New protein data TSV
--annotation         PATH    Optional   Annotation TSV (for reference)
--annotcol           STRING  Optional   Label column (if annotation provided)
--output             PATH    Required   Output directory
--classes            LIST    all        Filter to classes
```

---

## 📁 Expected Input Files

### Annotation File (TSV)
```
sample_id       multiomic_splitted
C3L.00104       nmf1
C3L.00365       nmf3
C3L.00674       nmf1
C3L.00677       nmf1
```

### Protein File (TSV)
```
                C3L.00104   C3L.00365   C3L.00674   C3L.00677
P04217          0.4         -0.550943   3.295082    -0.280230
P01023          2.259091    3.015094    4.418033    0.464491
Q9NRG9          -1.109091   0.694340    -0.459016   0.153551
```

---

## 📊 Output Files

### After TRAIN
```
xgb_model_<timestamp>.joblib          Model file (use in evaluate/apply)
best_params_<timestamp>.tsv            Best hyperparameters found
train_matrix_<timestamp>.tsv           Training data used
test_matrix_<timestamp>.tsv            Test data used
```

### After EVALUATE (OVR)
```
roc_curve_<timestamp>.png              Single ROC plot with all classes
evaluation_report_<timestamp>.tsv      Per-class precision/recall/f1
confusion_matrix_<timestamp>.tsv       Classification matrix
predicted_probabilities_<timestamp>.tsv Predictions ranked by confidence
predicted_samples_<timestamp>.png      Scatter plot of predictions
```

### After EVALUATE (OVO)
```
roc_pair_ClassA_vs_ClassB_<timestamp>.png    (One file per pair)
pairwise_aucs_<timestamp>.tsv                Pairwise AUC summary
evaluation_report_<timestamp>.tsv            Per-class metrics
confusion_matrix_<timestamp>.tsv             Classification matrix
predicted_probabilities_<timestamp>.tsv      Predictions
predicted_samples_<timestamp>.png            Prediction plot
```

### After APPLY
```
predicted_probabilities_<timestamp>_adhoc.tsv    Predictions on new data
predicted_samples_<timestamp>_adhoc.png          Confidence plot
```

---

## ⚡ Quick Examples

### Example 1: Train & Evaluate on GBM Data
```bash
# Train
python py_scripts/cli.py train \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output results \
  --n_iter 12 --init_points 3

# Evaluate with pairwise ROC
python py_scripts/cli.py evaluate \
  --model results/xgb_model_20260130114430.joblib \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output results \
  --roc_mode ovo
```

### Example 2: Train on CPTAC, Test on Werner
```bash
# Train on CPTAC
python py_scripts/cli.py train \
  --annotation ../GBM_testcase/CPTAC_annot.tsv \
  --protein ../GBM_testcase/CPTAC_data.tsv \
  --annotcol multiomic_splitted \
  --output results --n_iter 12

# Apply to Werner
python py_scripts/cli.py apply \
  --model results/xgb_model_*.joblib \
  --protein ../GBM_testcase/Werner_data.tsv \
  --annotation ../GBM_testcase/Werner_annot.tsv \
  --annotcol multiomic_splitted \
  --output results
```

### Example 3: Custom Data with Specific Classes
```bash
python py_scripts/cli.py train \
  --annotation my_annot.tsv \
  --protein my_proteins.tsv \
  --annotcol diagnosis \
  --output my_output \
  --classes Control Disease_A Disease_B \
  --n_iter 25 --init_points 5
```

---

## 🔧 Tuning Tips

| Goal | Settings |
|------|----------|
| Quick test | `--n_iter 3 --init_points 2` |
| Good balance | `--n_iter 12 --init_points 3` |
| Best performance | `--n_iter 25 --init_points 5` |
| Large dataset | `--testsize 0.2` |
| Small dataset | `--testsize 0.3` |
| Reproducible | `--seed 42` |

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Model file not found | Check timestamp matches. Use `dir output_dir` to list files |
| Column not found error | Verify `--annotcol` matches exactly (case-sensitive) |
| File not found error | Use absolute paths or run from multigroup directory |
| Small class warning | Expected if <3 samples in a class; results still work |
| OVO skips pairs | Pair has <10 samples; need more data for that pair |

---

## 📚 For More Details
See `WORKFLOW_GUIDE.md` for comprehensive explanations and troubleshooting.

