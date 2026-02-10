**ProteoBoostR (Python) — Training Report**

- **Scope**: This document explains how model training is implemented in the Python port under `py_scripts/`.

**Data Inputs**
- **Annotation file**: TSV with first column `sample_id` and one column chosen as the class label.
- **Protein matrix**: TSV where rows = protein IDs and columns = sample IDs (values numeric). This file is transposed and joined to the annotation on `sample_id`.

**Preprocessing**
- **File**: `py_scripts/utils.py`
- **Key functions**:
  - **`do_merge_static()`**: Transposes protein matrix, left-joins annotation by `sample_id`, cleans column names (removes trailing `;` parts), optional feature subset, converts feature columns to numeric and drops columns that are entirely NA.
  - **`preprocess_data()`**: Filters rows to remove NA labels, retains only the specified negative and positive classes, and converts the annotation column into a categorical factor with order `[neg, pos]`.

**Train/Test Split & Reproducibility**
- **Stratified split**: Uses sklearn `train_test_split(..., stratify=y)` to preserve class balance.
- **Parameters**: `--testsize` controls fraction held out for test; `--seed` sets random seed for reproducibility.
- **Saved artifacts**: `train_matrix_<ts>.tsv` and `test_matrix_<ts>.tsv` are written to the output directory.

**Hyperparameter Optimization (Bayesian)**
- **Location**: `py_scripts/cli.py` (the `train` subcommand).
- **Library**: `bayes_opt` (BayesianOptimization).
- **Search space (bounds)**:
  - `eta`: [0.01, 0.3]
  - `max_depth`: [1, 10]
  - `subsample`: [0.5, 1.0]
  - `colsample_bytree`: [0.5, 1.0]
  - `min_child_weight`: [1, 10]
  - `gamma`: [0.0, 5.0]
  - `alpha`: [0.0, 10.0]
  - `lambda`: [0.0, 10.0]
- **Control**: CLI flags `--init_points` and `--n_iter` control the number of initial random evaluations and optimization iterations (use small values for smoke tests).
- **Objective**: maximize 5-fold CV AUC computed via `xgboost.cv(...)`.
- **CV settings**: `nfold=5`, `num_boost_round=1000`, `early_stopping_rounds=50`, `eval_metric='auc'`.

**Final Model Training**
- After BO completes, the best parameter set is used to train a final XGBoost model on the training partition:
  - **Call**: `xgb.train(best_params, dtrain_full, num_boost_round=1000)`
  - Early stopping is not explicitly provided on the final `xgb.train` call in the current implementation (can be added by passing `early_stopping_rounds` and an evaluation set).
- **Saved artifacts**:
  - `xgb_model_<ts>.json` (trained model)
  - `best_params_<ts>.tsv` (tabular dump of final parameters)

**Evaluation (Test Set)**
- **Location**: `cli.py` `evaluate` subcommand.
- **Procedure**:
  - Load test set, build `xgb.DMatrix`, predict probabilities.
  - Compute ROC and AUC (`sklearn.metrics.roc_curve`, `auc`).
  - Determine best threshold using Youden index (closest to 0.5 if ties).
  - Compute confusion matrix, accuracy, sensitivity (recall), specificity.
  - Save `predicted_probabilities_<ts>.tsv`, `evaluation_results_<ts>.tsv`, `confusion_matrix_<ts>.tsv`, and `roc_curve_<ts>.png` to output folder.

**Ad-hoc Application (Independent Cohort)**
- **Location**: `py_scripts/apply_model.py` and `cli.py` `apply` subcommand.
- **Feature alignment**: Model feature names are obtained (helper `get_model_features`) and the new dataset is aligned to these features — missing features are filled with `NA` (numeric `NaN`), extra columns are ignored.
- **Threshold banding**: Optional base threshold can be supplied (TSV with `Best_Threshold`), and a band around it (`--band`) creates three zones — `below`, `not classified`, `above` — with predicted labels only for `below/above`.
- **Outputs**: Ranked prediction table and, if labels present and classified samples remain, confusion matrix + ROC for classified set.

**Logging & Errors**
- **Console logging**: `cli.py` uses a console logger for progress messages.
- **Per-run log file**: A file `proteoboostr_<ts>.log` is written into the output directory for training/evaluation/apply runs.
- **Exit codes**:
  - `2` — Missing input file or invalid output path
  - `3` — Model load or BO failure
  - `4` — Failure writing outputs
  - `1` — Generic unexpected error

**How to run (examples)**
- Quick train (short BO for smoke test):
```
python py_scripts/cli.py train --annotation py_scripts/GBM_testcase/Werner_annot.tsv \
  --protein py_scripts/GBM_testcase/Werner_data.tsv --annotcol subtypes \
  --neg 'subtype2+3+4' --pos subtype1 --output py_scripts/GBM_testcase/outputs_smoke \
  --testsize 0.3 --seed 123 --n_iter 2 --init_points 1
```
- Full training (increase `--n_iter` and `--init_points` as needed):
```
python py_scripts/cli.py train --annotation path/to/annot.tsv --protein path/to/protein.tsv \
  --annotcol YOUR_LABEL --neg NEG_VAL --pos POS_VAL --output outputs_folder --n_iter 20 --init_points 5
```
- Evaluate a saved model:
```
python py_scripts/cli.py evaluate --model outputs/xgb_model_<ts>.json --annotation path/to/annot.tsv \
  --protein path/to/protein.tsv --annotcol YOUR_LABEL --neg NEG_VAL --pos POS_VAL --output outputs_folder
```

**Notes, limitations & recommended improvements**
- **Final training early stopping**: Add an evaluation set and `early_stopping_rounds` to `xgb.train` to prevent overfitting and speed up training.
- **Single final model save**: Current runs saved multiple models during BO evaluation; adjust to save only the final best model (or save temporary evaluation models to a separate folder).
- **Feature name preservation**: Ensure the saved model stores feature names (the XGBoost booster should keep them if `DMatrix` column order matches); consider saving `feat_cols` alongside the model for robust alignment.
- **Hyperparameter search budget**: `--n_iter` and `--init_points` are exposed — use small values for testing and larger for production.
- **Logging verbosity**: Add a `--verbose` flag to control console/file log levels.
- **Unit tests**: Add tests for `do_merge_static`, `preprocess_data`, and for end-to-end train→evaluate flow on small synthetic datasets.

**Files of interest**
- **`py_scripts/cli.py`** — Unified CLI entry (`train`, `evaluate`, `apply`).
- **`py_scripts/utils.py`** — Preprocessing and helpers.
- **`py_scripts/apply_model.py`** — Standalone apply script (also accessible via CLI `apply`).
- **`py_scripts/requirements.txt`** — Python dependencies.
- **Outputs** — Saved to the `--output` folder you pass to the CLI; look for `xgb_model_<ts>.json`, `best_params_<ts>.tsv`, `*_matrix_<ts>.tsv`, `evaluation_results_*`, and `proteoboostr_<ts>.log`.

---

If you want, I can (pick one):
- Add final-training early stopping and save only one final model;
- Run `evaluate` on the model created in your smoke test and attach the evaluation TSV and ROC plot;
- Draft a quick test script that runs train→evaluate on a tiny synthetic dataset to validate end-to-end behavior.

