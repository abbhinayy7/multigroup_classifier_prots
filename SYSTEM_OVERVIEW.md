# ProteoBoostR System Overview

## Overview
ProteoBoostR is a **Shiny-based web application** for supervised classification in **proteomics data**. It's built around the XGBoost algorithm with Bayesian optimization for hyperparameter tuning.

---

## Architecture & Workflow

### 1. **Input Data**
#### File Formats (Tab-separated values, TSV):
- **Annotation File**: Contains metadata with sample IDs in the first column
  ```
  sample_id    subtypes
  TW688        subtype1
  TW689        subtype2+3+4
  TW690        subtype2+3+4
  ```

- **Protein Matrix**: Rows = protein IDs; Columns = sample IDs (expression values)
  ```
       TW688   TW689   TW690   TW691   ...
  A0A024RBG1  NA     NA     -0.466  NA
  A0A075B6H7  0.479  0.459  -0.326  NA
  ```

- **Protein Subset (Optional)**: List of protein IDs to filter (one per line)

#### Key Parameters:
- **Annotation Column**: Which column contains the class labels (excluded: sample_id)
- **Negative Class (0)** / **Positive Class (1)**: Define which values are neg/pos
- **Train/Test Split**: Percentage for partitioning data
- **Random Seed**: For reproducibility
- **Output Directory**: Where results are saved

### 2. **Data Preprocessing**
The `preprocessData()` function performs:
1. **Removes rows** with NA in the annotation column
2. **Filters to keep only** rows where annotation is negative OR positive class label
3. **Converts annotation column** to a factor with specified levels
4. **Merges** annotation and protein data (transpose protein matrix, join on sample_id)
5. **Removes ";" artifacts** from column names (e.g., "ProteinID;description" → "ProteinID")
6. **Converts feature columns** to numeric (drops columns that can't be converted)

### 3. **Model Training (XGBoost + Bayesian Optimization)**

#### Hyperparameter Tuning Process:
- **Algorithm**: Bayesian Optimization (BO)
- **BO Settings**: 5 initial points, 20 iterations, UCB acquisition
- **Evaluation**: 5-fold cross-validation with AUC as the metric

#### Tuned Parameters:
| Parameter | Default Range | Type |
|-----------|---------------|------|
| eta (learning rate) | [0.01, 0.3] | float |
| max_depth | [1, 10] | integer |
| subsample | [0.5, 1.0] | float |
| colsample_bytree | [0.5, 1.0] | float |
| min_child_weight | [1, 10] | integer |
| gamma | [0, 5] | float |
| alpha (L1 reg) | [0, 10] | float |
| lambda (L2 reg) | [0, 10] | float |

#### Fixed XGBoost Parameters:
- **booster**: "gbtree" (tree-based boosting)
- **objective**: "binary:logistic" (binary classification with probabilities)
- **eval_metric**: "auc" (Area Under the Curve)
- **nrounds**: 1000 (with early stopping at 50 rounds of no improvement)

#### Training Outputs:
- `xgb_model_<timestamp>.rds`: Serialized trained model
- `best_params_<timestamp>.tsv`: Best hyperparameters found
- `train_matrix_<timestamp>.tsv`: Preprocessed training data (transposed, merged)

### 4. **Model Testing (Evaluation)**

#### Evaluation Steps:
1. **Prediction**: Generate predicted probabilities on test set (0 to 1)
2. **ROC Curve**: Calculate sensitivity vs. 1-specificity
3. **Best Threshold**: Uses Youden's index to find optimal decision boundary
   - If multiple thresholds have same Youden value, selects closest to 0.5
4. **Classification**: Apply threshold to convert probabilities → class labels
5. **Metrics**: Calculate accuracy, sensitivity, specificity, AUC

#### Test Outputs:
- `predicted_probabilities_<timestamp>.tsv`: Scores for each sample
- `evaluation_results_<timestamp>.tsv`: Accuracy, Sensitivity, Specificity, AUC, Best_Threshold
- `confusion_matrix_<timestamp>.tsv`: True/False Positives/Negatives
- `roc_curve_<timestamp>.png`: ROC plot visualization
- `test_matrix_<timestamp>.tsv`: Preprocessed test data (transposed, merged)

#### Visualizations:
- **Ranked Prediction Plot**: Scatter plot of samples ranked by probability with threshold line
- **Score Table**: Ranked table with sample ID, annotation, predicted probability
- **Confusion Matrix**: Text output of TP, TN, FP, FN
- **ROC Curve**: PNG plot showing AUC value
- **Metrics**: Text output of performance statistics

### 5. **Model Application (Ad-hoc Testing)**

#### Capabilities:
- Apply trained model to **new, independent datasets**
- Work with or without **class labels**
- Supports **multiple parallel applications** via tabbed interface

#### Key Features:

1. **Feature Alignment**:
   - Model stores feature names internally
   - New data features are aligned to model's expected features
   - Missing features are filled with NA
   - Extra features are discarded

2. **Threshold Banding**:
   - Base threshold: From evaluation TSV or in-session model
   - ±Band: Creates uncertainty zone around threshold
   - **Three zones**:
     - **"above"**: Predictions > (threshold + band) → Positive class
     - **"not classified"**: Predictions within [threshold-band, threshold+band]
     - **"below"**: Predictions < (threshold - band) → Negative class

3. **Conditional Outputs**:
   - **Without labels**: Ranked probabilities table
   - **With labels**: Additionally outputs confusion matrix, ROC, metrics (only for classified samples)

#### Outputs (ad-hoc):
- `predicted_probabilities_<timestamp>_adhoc<N>.tsv`
- `evaluation_results_<timestamp>_adhoc<N>.tsv` (if labeled)
- `confusion_matrix_<timestamp>_adhoc<N>.tsv` (if labeled)
- All visualizations as PNG files

---

## UI Structure (Shiny Dashboard)

### Tabs:
1. **Landing**: Quick start guide
2. **Input for Training**: Upload files, set parameters
3. **Model Training**: Adjust BO parameters, start training
4. **Model Testing**: Evaluate trained model
5. **Model Application**: Apply model to new datasets (multiple tabs)
6. **Log**: Detailed processing messages

### Key UI Features:
- **Tabbed Interface**: Dynamic tab creation for multiple model applications
- **Progress Indicators**: Visual feedback during training/evaluation
- **Path Validation**: Checks if output directory exists (green ✓ or red ✗)
- **Help Text**: Context-specific hints on each input
- **Real-time Rendering**: Plots and tables update after evaluation

---

## Data Flow Diagram

```
Input Files (TSV)
    ↓
[Preprocessing]
    ├─ Merge annotation + protein matrix
    ├─ Filter to neg/pos classes only
    ├─ Remove NAs, clean column names
    └─ Convert features to numeric
    ↓
[Train/Test Split]
    ├─ Training Set → Model Training
    └─ Test Set → Model Evaluation
    ↓
[Bayesian Optimization] (5 init, 20 iter)
    ├─ 5-fold CV with XGBoost
    ├─ Maximize AUC
    └─ Find best hyperparameters
    ↓
[XGBoost Model Training]
    └─ Train on full training set with best params
    ↓
[Test Set Evaluation]
    ├─ Generate predictions (probabilities 0-1)
    ├─ Compute ROC & best threshold (Youden)
    ├─ Apply threshold → class labels
    └─ Calculate metrics (Accuracy, Sensitivity, Specificity, AUC)
    ↓
[Model Application] (Optional - Ad-hoc)
    ├─ New data preprocessing
    ├─ Feature alignment to model
    ├─ Predictions with threshold banding
    └─ Conditional evaluation (if labels present)
    ↓
Output Files (TSV + PNG)
```

---

## Test Case Example: GBM Study

### Data:
- **Werner cohort**: 56 samples, ~9700 proteins
- **CPTAC cohort**: (Application dataset)
- **Task**: Classify GBM subtypes (One vs. Rest)
- **Selected features**: 20 proteins from differential abundance analysis

### Results:
- Model trained on Werner with optimal hyperparameters
- Evaluated on 70/30 or 80/20 split
- Validated on CPTAC as independent dataset

---

## Key Technical Details

### Python Equivalents:
- **R Data Frame** ↔ **pandas DataFrame**
- **xgboost (R)** ↔ **xgboost (Python)**
- **caret::confusionMatrix** ↔ **sklearn.metrics.confusion_matrix**
- **pROC::roc** ↔ **sklearn.metrics.roc_curve** + **auc**

### Dependencies:
- **shiny, shinydashboard, shinyjs**: UI framework
- **DT**: Interactive data tables
- **dplyr, tibble**: Data manipulation
- **caret**: Evaluation metrics
- **xgboost**: ML algorithm
- **pROC**: ROC curve analysis
- **rBayesianOptimization**: Hyperparameter tuning
- **ggplot2**: Visualization
- **fontawesome**: Icons

### Special Notes:
- Docker mode enforces folder names only (no slashes) in output path
- Local mode accepts full paths
- File uploads limited to 100 MB per file
- Early stopping in XGBoost: 50 rounds with no AUC improvement
- Threshold selection prefers Youden-optimal value closest to 0.5

---

## Common Use Cases

1. **Discover Proteomics Biomarkers**:
   - Train on discovery cohort (e.g., Werner)
   - Test on held-out samples
   - Apply to independent cohort (e.g., CPTAC)

2. **Disease Subtype Classification**:
   - Binary classification of disease subtypes
   - Generate ranked predictions per subtype
   - Identify uncertain cases (threshold band)

3. **Reproducible ML Pipelines**:
   - Fixed seed for train/test split
   - Saved hyperparameters for model versioning
   - Full audit trail in log output
