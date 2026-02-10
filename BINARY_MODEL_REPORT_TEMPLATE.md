# Binary Classification Model Report Template
## ProteoBoostR XGBoost Analysis

**Report Date:** [DATE]  
**Dataset:** [DATASET NAME]  
**Model ID:** [MODEL TIMESTAMP]  
**Classification Task:** [POSITIVE CLASS] vs [NEGATIVE CLASS]

---

## Executive Summary

This report documents the development and performance of an XGBoost binary classification model trained on proteomics data to distinguish between [NEGATIVE CLASS] and [POSITIVE CLASS] samples.

### Key Findings at a Glance
- **Model Performance:** [ACCURACY]% accuracy on test set
- **Discriminative Power:** AUC = [AUC_VALUE] (0=random, 1=perfect)
- **Primary Metric:** [BEST_METRIC] at optimal threshold of [THRESHOLD]
- **Sample Size:** [N_TRAINING] training samples, [N_TEST] test samples
- **Features Used:** [N_FEATURES] protein expression measurements

---

## 1. Background & Objectives

### 1.1 Scientific Question
What proteomics features can reliably predict whether a sample belongs to the [POSITIVE CLASS] or [NEGATIVE CLASS] category?

### 1.2 Biological Significance
- **Positive Class ([POSITIVE CLASS]):** [DESCRIPTION]
- **Negative Class ([NEGATIVE CLASS]):** [DESCRIPTION]
- **Clinical Relevance:** [EXPLAIN WHY THIS CLASSIFICATION MATTERS]

### 1.3 Model Purpose
This model can:
- ✓ Predict class membership for new samples based on protein expression
- ✓ Identify proteomics signatures that differentiate the two groups
- ✓ Support clinical decision-making or further research [AS APPLICABLE]

---

## 2. Methods

### 2.1 Data Source & Preparation

#### Input Data
| Aspect | Details |
|--------|---------|
| **Annotation File** | [FILE NAME / PATH] |
| **Sample Size (Raw)** | [N] total samples |
| **Features (Raw)** | [N] protein features |
| **Class Distribution** | [N/PERCENT] positive, [N/PERCENT] negative |

#### Preprocessing Steps
1. **Data Cleaning**
   - Removed samples with missing class labels: [N] samples
   - Kept only positive/negative class samples: excluded [N] samples
   - Removed proteins with >80% missing values: [N] proteins dropped

2. **Feature Standardization**
   - All protein expression values converted to numeric scale
   - Missing values (NA) handled with feature-wise filtering
   - Non-convertible features removed: [N] features

3. **Sample Filtering**
   - Final training dataset: [N_TRAIN] samples × [N_FEATURES] features
   - Final test dataset: [N_TEST] samples × [N_FEATURES] features
   - Train/test split: [SPLIT_RATIO] (stratified by class)

### 2.2 Machine Learning Approach

#### Algorithm Selection: XGBoost
```
Why XGBoost for Proteomics?
├─ Handles high-dimensional data (1000s of features)
├─ Robust to missing values
├─ Captures non-linear interactions between proteins
├─ Provides feature importance rankings
├─ Produces calibrated probability estimates
└─ Fast training on modern hardware
```

#### Hyperparameter Optimization
**Method:** Bayesian Optimization with Gaussian Processes
- **Objective:** Maximize cross-validation AUC
- **CV Strategy:** 5-fold stratified cross-validation
- **Search Budget:** [N_ITER] iterations ([INIT_POINTS] random + [N_ITER-INIT_POINTS] Bayesian)
- **Acquisition Function:** Upper Confidence Bound (UCB)

#### Optimized Hyperparameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **eta** (learning rate) | [VALUE] | How quickly the model learns (0-1; lower=slower/safer) |
| **max_depth** | [VALUE] | Maximum tree depth (how complex each tree can be) |
| **subsample** | [VALUE] | Fraction of samples used per tree (0-1; prevents overfitting) |
| **colsample_bytree** | [VALUE] | Fraction of features per tree (0-1) |
| **min_child_weight** | [VALUE] | Minimum samples per leaf (prevents overfitting) |
| **gamma** | [VALUE] | Minimum loss reduction needed for split |
| **alpha** (L1) | [VALUE] | L1 regularization strength (feature selection) |
| **lambda** (L2) | [VALUE] | L2 regularization strength (weight penalty) |

#### Fixed Parameters
```
booster = gbtree                    (gradient boosting with trees)
objective = binary:logistic         (binary classification logits)
eval_metric = auc                   (optimize for AUC)
nrounds = 1500                      (max boosting iterations)
early_stopping_rounds = 100         (stop if no improvement for 100 rounds)
```

### 2.3 Model Validation Strategy

**Cross-Validation During Training:**
- 5-fold stratified split maintains class distribution
- Each fold: 80% training, 20% validation
- Final metrics reported on held-out test set

**Independence of Test Set:**
- Test samples not used in hyperparameter optimization
- Provides unbiased performance estimate
- Simulates real-world deployment scenario

---

## 3. Results & Model Performance

### 3.1 Classification Performance

#### Confusion Matrix (Test Set)
```
                    Predicted Negative    Predicted Positive
Actual Negative:           [TN]                  [FP]
Actual Positive:           [FN]                  [TP]
```

#### Primary Performance Metrics

| Metric | Definition | Value | Interpretation |
|--------|-----------|-------|-----------------|
| **Accuracy** | (TP+TN) / Total | [VALUE]% | Overall correctness |
| **Sensitivity (TPR)** | TP / Positives | [VALUE]% | Catches positives; avoid false negatives |
| **Specificity (TNR)** | TN / Negatives | [VALUE]% | Avoids false positives; confidence in negative calls |
| **Precision (PPV)** | TP / Predicted Pos | [VALUE]% | When predicting positive, how often correct |
| **F1-Score** | 2×Precision×Recall/(Precision+Recall) | [VALUE] | Balance of precision & recall |

#### Advanced Metrics

| Metric | Value | Significance |
|--------|-------|--------------|
| **AUC (Area Under Curve)** | [VALUE] | Overall discriminative ability (0=random, 1=perfect) |
| **Best Threshold** | [VALUE] | Optimal decision boundary (Youden's J index) |
| **Matthews Correlation Coefficient** | [VALUE] | Correlation between predicted and actual (-1 to 1) |

#### Performance Interpretation
**What these numbers mean:**
- If **Sensitivity is high**: Model catches most true positives (good for screening)
- If **Specificity is high**: Model avoids false alarms (good for confirmation)
- If **Precision is high**: Positive predictions are reliable (few false positives)
- If **AUC > 0.90**: Excellent discrimination between classes
- If **AUC > 0.70**: Good discrimination (often acceptable)
- If **AUC ≤ 0.50**: Model performs like random guessing

### 3.2 ROC Curve Analysis

**Location:** `roc_curve_[TIMESTAMP].png`

```
What the ROC Curve Shows:
- X-axis: False Positive Rate (1 - Specificity)
  [Going right = more false alarms]
  
- Y-axis: True Positive Rate (Sensitivity)
  [Going up = better detection of positives]

- Red Dot: Operating Point at threshold [VALUE]
  [Best balance at this decision threshold]

- Green Line: Random Classifier (AUC = 0.50)
  [If your curve touches this line, model isn't working]

- Blue Area: AUC = [VALUE]
  [Higher area = better discrimination]
```

**Interpreting Your Curve:**
- **Closer to top-left corner:** Model is excellent
- **Closer to diagonal line:** Model is poor
- **Threshold [VALUE] represents:** The probability cutoff for classifying as positive

### 3.3 Sample Predictions & Confidence

**Location:** `predicted_samples_[TIMESTAMP].png`

```
What This Plot Shows:
- X-axis: Samples ranked by predicted probability (highest → lowest)
- Y-axis: Predicted probability (0=definitely negative, 1=definitely positive)
- Blue dots: Actual negative samples
- Red dots: Actual positive samples
- Green line: Decision threshold at [VALUE]

Interpretation:
- Points above green line: Predicted as positive
- Points below green line: Predicted as negative
- Clear separation: Model is confident
- Overlapping dots: Model is uncertain
```

**Sample Predictions Table:**
```
Sample_ID | Actual_Class | Predicted_Prob | Predicted_Class | Confidence | Correct?
----------|--------------|----------------|-----------------|------------|----------
[ID1]     | [ACTUAL]     | [PROB]         | [PRED]           | [CONF]%    | ✓/✗
[ID2]     | [ACTUAL]     | [PROB]         | [PRED]           | [CONF]%    | ✓/✗
[ID3]     | [ACTUAL]     | [PROB]         | [PRED]           | [CONF]%    | ✓/✗
```

### 3.4 Feature Importance

**Top 10 Most Predictive Proteins:**

| Rank | Protein ID | Importance Score | Direction | Interpretation |
|------|-----------|------------------|-----------|-----------------|
| 1 | [PROTEIN] | [SCORE] | ↑ Higher in Positive | Key marker |
| 2 | [PROTEIN] | [SCORE] | ↓ Lower in Positive | Key marker |
| 3 | [PROTEIN] | [SCORE] | Mixed | Complex pattern |
| ... | ... | ... | ... | ... |
| 10 | [PROTEIN] | [SCORE] | ↑ Higher in Positive | Secondary marker |

**How to Read Feature Importance:**
- **High scores:** Proteins that strongly influence predictions
- **↑ Higher in Positive:** Elevated in [POSITIVE CLASS] samples
- **↓ Lower in Positive:** Reduced in [POSITIVE CLASS] samples
- **Mixed:** Complex non-linear relationship

---

## 4. Model Quality & Reliability

### 4.1 Overfitting Assessment

```
Training Performance:    [TRAIN_ACCURACY]% accuracy
Test Performance:        [TEST_ACCURACY]% accuracy
Difference:              [DIFF]%

Interpretation:
├─ < 5% difference: ✓ Good generalization
├─ 5-10% difference: ⚠ Slight overfitting
└─ > 10% difference: ⚠ Significant overfitting
```

**Regularization Applied:**
- L1 regularization (alpha): [VALUE] - Encourages feature selection
- L2 regularization (lambda): [VALUE] - Penalizes large weights
- Early stopping: Stops at 100 rounds without improvement
- Subsampling: [VALUE]% of samples per tree

### 4.2 Cross-Validation Results

**5-Fold CV Performance (Training Process):**

| Fold | AUC | Accuracy | Sensitivity | Specificity |
|-----|-----|----------|-------------|-------------|
| Fold 1 | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% |
| Fold 2 | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% |
| Fold 3 | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% |
| Fold 4 | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% |
| Fold 5 | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% |
| **Mean ± SD** | **[MEAN] ± [SD]** | **[MEAN]% ± [SD]%** | ... | ... |

**Consistency Check:**
- Low SD (< 5%): Model is stable and reliable
- High SD (> 10%): Model may be sensitive to dataset variation

### 4.3 Threshold Sensitivity

**How performance changes at different thresholds:**

| Threshold | Sensitivity | Specificity | Accuracy | Youden's J |
|-----------|-------------|-------------|----------|-----------|
| 0.30 | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| 0.50 | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [OPTIMAL] | [VALUE]% | [VALUE]% | [VALUE]% | [OPTIMAL] |
| 0.70 | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| 0.90 | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |

**Choosing Your Threshold:**
- **Want to catch all positives:** Use lower threshold (0.3-0.5) - Higher sensitivity
- **Want to minimize false alarms:** Use higher threshold (0.7-0.9) - Higher specificity
- **Balanced decision:** Use optimal threshold [VALUE] - Best overall performance

---

## 5. Data Characteristics

### 5.1 Class Distribution

```
Class Distribution:
Negative [CLASS_NAME]:    [N] samples ([PERCENT]%)
Positive [CLASS_NAME]:    [N] samples ([PERCENT]%)
─────────────────────────────────────
Total:                    [N] samples

Balance Assessment:
├─ < 1:2 ratio: Balanced ✓
├─ 1:2 - 1:5 ratio: Slightly imbalanced
└─ > 1:5 ratio: Highly imbalanced ⚠
```

### 5.2 Feature Characteristics

| Aspect | Value |
|--------|-------|
| **Total Features** | [N] proteins |
| **Feature Range** | [MIN] to [MAX] |
| **Missing Data** | [PERCENT]% missing values |
| **Numeric Conversion** | [N] features successfully converted |
| **Features Removed** | [N] features (non-numeric) |

---

## 6. Clinical/Practical Implications

### 6.1 Model Deployment

**When to use this model:**
✓ [USE CASE 1]
✓ [USE CASE 2]  
✓ [USE CASE 3]

**When NOT to use this model:**
✗ [LIMITATION 1]
✗ [LIMITATION 2]
✗ [LIMITATION 3]

### 6.2 Decision Framework

```
For a new sample with predicted probability P:

If P > [THRESHOLD]:
  Decision: Predict [POSITIVE CLASS]
  Confidence: [CONF_HIGH]%
  Further Action: [RECOMMENDED_ACTION]

If P < [THRESHOLD]:
  Decision: Predict [NEGATIVE CLASS]
  Confidence: [CONF_HIGH]%
  Further Action: [RECOMMENDED_ACTION]

Borderline Cases (P ≈ [THRESHOLD]):
  Recommendation: Review top features, consider repeat testing
```

### 6.3 Important Caveats

⚠️ **Model Limitations:**
1. **Limited to [POSITIVE CLASS] vs [NEGATIVE CLASS]:** Cannot predict other classes
2. **Training data specific:** Performance may differ on new populations
3. **Protein expression dependent:** Requires same measurement platform
4. **Binary output:** Provides probability, not biological interpretation
5. **Size constraint:** Trained on [N_TRAIN] samples
6. **Missing data:** Handles missing values by exclusion

⚠️ **When Model May Fail:**
- On samples from different tissue types
- With different proteomics platforms
- On highly imbalanced samples
- With systematically different protein distributions

---

## 7. Reproducibility & Implementation

### 7.1 Files Generated

```
├── xgb_model_[TIMESTAMP].json
│   └─ Trained model (binary format, ready for predictions)
├── best_params_[TIMESTAMP].tsv
│   └─ Optimal hyperparameters from Bayesian optimization
├── train_matrix_[TIMESTAMP].tsv
│   └─ Training data (rows=samples, cols=proteins)
├── test_matrix_[TIMESTAMP].tsv
│   └─ Test data (rows=samples, cols=proteins)
├── predicted_probabilities_[TIMESTAMP].tsv
│   └─ Probability scores for each test sample
├── evaluation_results_[TIMESTAMP].tsv
│   └─ All metrics (accuracy, AUC, threshold, etc.)
├── confusion_matrix_[TIMESTAMP].tsv
│   └─ TP/TN/FP/FN breakdown
├── roc_curve_[TIMESTAMP].png
│   └─ ROC curve visualization
├── predicted_samples_[TIMESTAMP].png
│   └─ Ranked samples visualization
└── proteoboostr_[TIMESTAMP].log
    └─ Full training execution log
```

### 7.2 Making Predictions on New Data

```bash
python py_scripts/cli.py apply \
  --model path/to/xgb_model_[TIMESTAMP].json \
  --protein new_protein_data.tsv \
  --annotation new_annotation.tsv \
  --annotcol class_column_name \
  --neg negative_label \
  --pos positive_label \
  --evaltsv path/to/evaluation_results_[TIMESTAMP].tsv \
  --output predictions_folder/
```

**Output:** Probability scores + rankings + visualizations for new samples

---

## 8. Comparison to Baseline

### 8.1 Performance Improvement

| Method | Accuracy | AUC | Advantage |
|--------|----------|-----|-----------|
| **XGBoost (This Model)** | **[VALUE]%** | **[VALUE]** | ✓ Best overall |
| Random Classifier | 50% | 0.50 | Baseline |
| [Other Method] | [VALUE]% | [VALUE] | [COMPARISON] |

### 8.2 Why This Model Performs Well

✓ **Bayesian Optimization:** Systematically found best hyperparameters
✓ **Feature Engineering:** Uses raw protein expressions (validated signal)
✓ **Cross-Validation:** Avoids overfitting through rigorous testing
✓ **Regularization:** L1/L2 penalties prevent model complexity
✓ **Stratified Sampling:** Maintains class balance in train/test splits

---

## 9. Future Improvements & Next Steps

### 9.1 Potential Enhancements
- [ ] Increase training dataset size (currently [N_TRAIN] samples)
- [ ] Test on independent validation cohort
- [ ] Perform external validation with [COLLABORATING_LAB]
- [ ] Incorporate clinical covariates (age, sex, stage)
- [ ] Analyze feature interactions with domain experts
- [ ] Test portability across proteomics platforms

### 9.2 Recommended Next Steps
1. **Short-term:** [ACTION]
2. **Medium-term:** [ACTION]
3. **Long-term:** [ACTION]

---

## 10. Technical Specifications

### 10.1 Software & Versions
```
Python: 3.8+
XGBoost: 1.7+
scikit-learn: 1.0+
Pandas: 1.3+
Matplotlib: 3.4+
bayesian-optimization: 1.2+
```

### 10.2 Computational Requirements
- **Training Time:** [DURATION]
- **Features:** [N_FEATURES] proteins
- **Samples:** [N_SAMPLES] total
- **Memory:** ~[MB] RAM
- **Model File Size:** [SIZE] KB

### 10.3 Hyperparameter Search Details
```
Bayesian Optimization Log:
Iteration 1:  AUC = 0.xxx (random)
Iteration 2:  AUC = 0.xxx (random)
...
Iteration 15: AUC = 0.xxx (Bayesian) ✓ Best found
```

---

## 11. References & Methodology

### 11.1 Key Publications
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Youden, W. J. (1950). Index for rating diagnostic tests. Cancer, 3(1), 32-35.
- Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874.

### 11.2 Model Card Summary
```
Model Name:        ProteoBoostR Binary Classifier
Task:              Binary classification (Proteomics)
Algorithm:         XGBoost with Bayesian optimization
Input:             Protein expression matrix
Output:            Binary class + probability score
Training Data:     [N_TRAIN] samples, [N_FEATURES] features
Test Data:         [N_TEST] samples
Performance:       [ACCURACY]% accuracy, AUC=[AUC]
Intended Use:      [USE CASE]
Not for:           [CONTRAINDICATION]
```

---

## 12. Questions & Troubleshooting

### Q: What does the probability score mean?
**A:** It's the model's confidence that the sample belongs to [POSITIVE CLASS]. Score of 0.90 means 90% confident in positive, 10% confident in negative.

### Q: Can I change the decision threshold?
**A:** Yes! The threshold [VALUE] is optimal, but you can adjust it:
- Lower threshold → catches more positives (higher sensitivity)
- Higher threshold → fewer false alarms (higher specificity)

### Q: What if a sample has missing proteins?
**A:** The model was trained on complete cases. Samples with missing values in key features may have unreliable predictions.

### Q: How often should the model be retrained?
**A:** Ideally when:
- New large datasets become available ([N_MIN_SAMPLES]+ samples)
- Proteomics platform changes
- Class distribution shifts significantly
- Performance drops below threshold

### Q: Can this model predict [OTHER_CLASS]?
**A:** No, this model is trained only for [POSITIVE CLASS] vs [NEGATIVE CLASS]. A separate model is needed for additional classes.

---

## 13. Approval & Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Model Developer | [NAME] | __________ | __________ |
| Reviewer | [NAME] | __________ | __________ |
| Approver | [NAME] | __________ | __________ |

---

## Appendix A: Detailed Evaluation Metrics

**Contingency Table:**
```
               Predicted Negative  Predicted Positive  Total
Negative       TN=[VALUE]          FP=[VALUE]          [VALUE]
Positive       FN=[VALUE]          TP=[VALUE]          [VALUE]
Total          [VALUE]             [VALUE]             [VALUE]
```

**Derived Metrics:**
- Sensitivity = TP / (TP + FN) = [VALUE]%
- Specificity = TN / (TN + FP) = [VALUE]%
- Precision = TP / (TP + FP) = [VALUE]%
- NPV (Neg Pred Val) = TN / (TN + FN) = [VALUE]%
- Youden's J = Sensitivity + Specificity - 1 = [VALUE]

---

**Report Generated:** [DATE TIME]  
**Analyst:** [NAME]  
**Institution:** [ORG]  
**Contact:** [EMAIL]
