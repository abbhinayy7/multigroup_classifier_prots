#!/bin/bash
# ============================================================================
# MultiGroup ProteoBoostR - Complete Workflow Script (Bash/Linux/Mac)
# ============================================================================
# This script demonstrates the full multiclass proteomics classification pipeline:
# 1. Generate synthetic test data (optional)
# 2. Train XGBoost model with Bayesian hyperparameter optimization
# 3. Evaluate model with One-vs-Rest (OVR) ROC
# 4. Evaluate model with One-vs-One (OVO) pairwise ROC
# 5. Apply model to independent test dataset
# ============================================================================

set -e  # Exit on any error

CLI="./py_scripts/cli.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MultiGroup ProteoBoostR - Full Workflow"
echo "=========================================="

# ============================================================================
# TASK 1: Generate Synthetic Test Data (Optional Demo)
# ============================================================================
echo -e "\n[1/5] Generating synthetic multiclass dataset..."

python3 test_data/generate_synthetic.py

# ============================================================================
# TASK 2: Train Model on Synthetic Data
# ============================================================================
echo -e "\n[2/5] Training model on synthetic data..."

python3 $CLI train \
  --annotation test_data/annotation.tsv \
  --protein test_data/protein.tsv \
  --annotcol label \
  --output test_output \
  --n_iter 3 --init_points 2 --testsize 0.2 --seed 42

echo "✓ Model trained. Outputs saved to: test_output"

# ============================================================================
# TASK 3: Evaluate on Synthetic Data - One-vs-Rest (OVR) ROC
# ============================================================================
echo -e "\n[3/5] Evaluating model - One-vs-Rest (OVR) ROC..."

# Get the latest model file
MODEL_FILE=$(ls -t test_output/xgb_model_*.joblib 2>/dev/null | head -1)

python3 $CLI evaluate \
  --model $MODEL_FILE \
  --annotation test_data/annotation.tsv \
  --protein test_data/protein.tsv \
  --annotcol label \
  --output test_output \
  --roc_mode ovr

echo "✓ OVR evaluation complete. Check output directory for ROC curve PNG"

# ============================================================================
# TASK 4: Evaluate on Synthetic Data - One-vs-One (OVO) Pairwise ROC
# ============================================================================
echo -e "\n[4/5] Evaluating model - One-vs-One (OVO) Pairwise ROC..."

python3 $CLI evaluate \
  --model $MODEL_FILE \
  --annotation test_data/annotation.tsv \
  --protein test_data/protein.tsv \
  --annotcol label \
  --output test_output \
  --roc_mode ovo

echo "✓ OVO evaluation complete. Pairwise ROC PNGs saved"

# ============================================================================
# TASK 5: Apply Model to Independent Data (Ad-hoc Prediction)
# ============================================================================
echo -e "\n[5/5] Applying model to independent dataset (ad-hoc)..."

python3 $CLI apply \
  --model $MODEL_FILE \
  --protein test_data/protein.tsv \
  --annotation test_data/annotation.tsv \
  --annotcol label \
  --output test_output

echo "✓ Ad-hoc prediction complete"

# ============================================================================
# Summary
# ============================================================================
echo -e "\n=========================================="
echo "All Tasks Completed Successfully!"
echo "=========================================="
echo -e "\nOutput Directory: $(pwd)/test_output"
echo -e "\nKey Output Files:"
echo "  - xgb_model_<timestamp>.joblib          [Trained model]"
echo "  - best_params_<timestamp>.tsv           [Hyperparameters]"
echo "  - evaluation_report_<timestamp>.tsv     [Per-class metrics]"
echo "  - confusion_matrix_<timestamp>.tsv      [Classification matrix]"
echo "  - roc_curve_<timestamp>.png             [One-vs-Rest ROC]"
echo "  - roc_pair_*_vs_*_<timestamp>.png       [Pairwise ROC curves]"
echo "  - pairwise_aucs_<timestamp>.tsv         [Pairwise AUC summary]"
echo "  - predicted_samples_<timestamp>.png     [Ranked prediction plot]"
echo "  - predicted_probabilities_<timestamp>.tsv [Predictions with scores]"
echo "=========================================="
