# ============================================================================
# MultiGroup ProteoBoostR - Complete Workflow Script
# ============================================================================
# This script demonstrates the full multiclass proteomics classification pipeline:
# 1. Generate synthetic test data (optional)
# 2. Train XGBoost model with Bayesian hyperparameter optimization
# 3. Evaluate model with One-vs-Rest (OVR) ROC
# 4. Evaluate model with One-vs-One (OVO) pairwise ROC
# 5. Apply model to independent test dataset
# ============================================================================

Write-Host "=========================================="
Write-Host "MultiGroup ProteoBoostR - Full Workflow"
Write-Host "=========================================="

$CLI = "f:\ProteoBoostR\multigroup\py_scripts\cli.py"
$SCRIPT_DIR = "f:\ProteoBoostR\multigroup"

# ============================================================================
# TASK 1: Generate Synthetic Test Data (Optional Demo)
# ============================================================================
Write-Host "`n[1/5] Generating synthetic multiclass dataset..." -ForegroundColor Cyan

python "$SCRIPT_DIR\test_data\generate_synthetic.py"

# ============================================================================
# TASK 2: Train Model on Synthetic Data
# ============================================================================
Write-Host "`n[2/5] Training model on synthetic data..." -ForegroundColor Cyan

python $CLI train `
  --annotation "$SCRIPT_DIR\test_data\annotation.tsv" `
  --protein "$SCRIPT_DIR\test_data\protein.tsv" `
  --annotcol label `
  --output "$SCRIPT_DIR\test_output" `
  --n_iter 3 --init_points 2 --testsize 0.2 --seed 42

Write-Host "✓ Model trained. Outputs saved to: $SCRIPT_DIR\test_output" -ForegroundColor Green

# ============================================================================
# TASK 3: Evaluate on Synthetic Data - One-vs-Rest (OVR) ROC
# ============================================================================
Write-Host "`n[3/5] Evaluating model - One-vs-Rest (OVR) ROC..." -ForegroundColor Cyan

# Get the latest model file
$MODEL_FILE = Get-ChildItem "$SCRIPT_DIR\test_output" -Filter "xgb_model_*.joblib" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }

python $CLI evaluate `
  --model $MODEL_FILE `
  --annotation "$SCRIPT_DIR\test_data\annotation.tsv" `
  --protein "$SCRIPT_DIR\test_data\protein.tsv" `
  --annotcol label `
  --output "$SCRIPT_DIR\test_output" `
  --roc_mode ovr

Write-Host "✓ OVR evaluation complete. Check output directory for ROC curve PNG" -ForegroundColor Green

# ============================================================================
# TASK 4: Evaluate on Synthetic Data - One-vs-One (OVO) Pairwise ROC
# ============================================================================
Write-Host "`n[4/5] Evaluating model - One-vs-One (OVO) Pairwise ROC..." -ForegroundColor Cyan

python $CLI evaluate `
  --model $MODEL_FILE `
  --annotation "$SCRIPT_DIR\test_data\annotation.tsv" `
  --protein "$SCRIPT_DIR\test_data\protein.tsv" `
  --annotcol label `
  --output "$SCRIPT_DIR\test_output" `
  --roc_mode ovo

Write-Host "✓ OVO evaluation complete. Pairwise ROC PNGs saved" -ForegroundColor Green

# ============================================================================
# TASK 5: Apply Model to Independent Data (Ad-hoc Prediction)
# ============================================================================
Write-Host "`n[5/5] Applying model to independent dataset (ad-hoc)..." -ForegroundColor Cyan

python $CLI apply `
  --model $MODEL_FILE `
  --protein "$SCRIPT_DIR\test_data\protein.tsv" `
  --annotation "$SCRIPT_DIR\test_data\annotation.tsv" `
  --annotcol label `
  --output "$SCRIPT_DIR\test_output"

Write-Host "✓ Ad-hoc prediction complete" -ForegroundColor Green

# ============================================================================
# BONUS: Real Data Example - GBM CPTAC Dataset
# ============================================================================
Write-Host "`n[BONUS] Training on real GBM CPTAC data..." -ForegroundColor Yellow

$GBM_ANNOT = "f:\ProteoBoostR\GBM_testcase\CPTAC_annot.tsv"
$GBM_DATA = "f:\ProteoBoostR\GBM_testcase\CPTAC_data.tsv"
$GBM_OUTPUT = "f:\ProteoBoostR\GBM_testcase\multigroup_output"

# Train
Write-Host "Training on GBM CPTAC (3 classes: nmf1, nmf2, nmf3)..." -ForegroundColor Cyan
python $CLI train `
  --annotation $GBM_ANNOT `
  --protein $GBM_DATA `
  --annotcol multiomic_splitted `
  --output $GBM_OUTPUT `
  --n_iter 12 --init_points 3 --testsize 0.25 --seed 42

Write-Host "✓ GBM training complete" -ForegroundColor Green

# Get latest GBM model
$GBM_MODEL = Get-ChildItem $GBM_OUTPUT -Filter "xgb_model_*.joblib" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }

# Evaluate with OVO
Write-Host "Evaluating GBM model with OVO ROC..." -ForegroundColor Cyan
python $CLI evaluate `
  --model $GBM_MODEL `
  --annotation $GBM_ANNOT `
  --protein $GBM_DATA `
  --annotcol multiomic_splitted `
  --output $GBM_OUTPUT `
  --roc_mode ovo

Write-Host "✓ GBM evaluation complete" -ForegroundColor Green

# Apply to Werner data
Write-Host "Applying GBM model to Werner dataset..." -ForegroundColor Cyan
$WERNER_ANNOT = "f:\ProteoBoostR\GBM_testcase\Werner_annot.tsv"
$WERNER_DATA = "f:\ProteoBoostR\GBM_testcase\Werner_data.tsv"

python $CLI apply `
  --model $GBM_MODEL `
  --protein $WERNER_DATA `
  --annotation $WERNER_ANNOT `
  --annotcol multiomic_splitted `
  --output $GBM_OUTPUT

Write-Host "✓ GBM apply complete - Werner predictions saved" -ForegroundColor Green

# ============================================================================
# Summary
# ============================================================================
Write-Host "`n=========================================="
Write-Host "All Tasks Completed Successfully!"
Write-Host "=========================================="
Write-Host "`nOutput Directories:"
Write-Host "  Synthetic test: $SCRIPT_DIR\test_output"
Write-Host "  GBM CPTAC:      $GBM_OUTPUT"
Write-Host "`nKey Output Files:"
Write-Host "  - xgb_model_<timestamp>.joblib          [Trained model]"
Write-Host "  - best_params_<timestamp>.tsv           [Hyperparameters]"
Write-Host "  - evaluation_report_<timestamp>.tsv     [Per-class metrics]"
Write-Host "  - confusion_matrix_<timestamp>.tsv      [Classification matrix]"
Write-Host "  - roc_curve_<timestamp>.png             [One-vs-Rest ROC]"
Write-Host "  - roc_pair_*_vs_*_<timestamp>.png       [Pairwise ROC curves]"
Write-Host "  - pairwise_aucs_<timestamp>.tsv         [Pairwise AUC summary]"
Write-Host "  - predicted_samples_<timestamp>.png     [Ranked prediction plot]"
Write-Host "  - predicted_probabilities_<timestamp>.tsv [Predictions with scores]"
Write-Host "=========================================="
