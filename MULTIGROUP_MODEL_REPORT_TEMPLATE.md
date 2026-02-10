# Multigroup Classification Model Report Template
## ProteoBoostR XGBoost Multiclass Analysis

**Report Date:** [DATE]  
**Dataset:** [DATASET NAME]  
**Model ID:** [MODEL TIMESTAMP]  
**Classification Task:** [NUMBER] Classes: [CLASS1] vs [CLASS2] vs [CLASS3] vs ...

---

## Executive Summary

This report documents the development and performance of an XGBoost multiclass classification model trained on proteomics data to distinguish between [N_CLASSES] distinct sample groups: [LIST_CLASSES].

### Key Findings at a Glance
- **Model Type:** Multiclass XGBoost with [STRATEGY] strategy
- **Number of Classes:** [N] classes
- **Overall Accuracy:** [ACCURACY]% (correctly classified across all classes)
- **Macro-Average AUC:** [AUC_MACRO] (unweighted average across all pairs)
- **Sample Distribution:** [CLASS1]: [N] ([%]), [CLASS2]: [N] ([%]), ...
- **Features Used:** [N_FEATURES] protein expression measurements
- **Pairwise Comparisons:** [N_PAIRS] One-vs-One (OVO) ROC curves generated

---

## 1. Background & Objectives

### 1.1 Scientific Question
Can proteomics features reliably distinguish among [N_CLASSES] sample groups: [CLASS1], [CLASS2], [CLASS3], etc.?

### 1.2 Class Definitions

| Class | Label | Description | Biology |
|-------|-------|-------------|---------|
| [CLASS1] | [LABEL1] | [DESC1] | [BIO1] |
| [CLASS2] | [LABEL2] | [DESC2] | [BIO2] |
| [CLASS3] | [LABEL3] | [DESC3] | [BIO3] |
| [CLASS4] | [LABEL4] | [DESC4] | [BIO4] |

### 1.3 Clinical Relevance
This classification model supports:
✓ [USE CASE 1]
✓ [USE CASE 2]
✓ [USE CASE 3]

### 1.4 Model Purpose
This model can:
- Predict class membership for new samples based on protein profiles
- Identify proteins that differentiate the multiple groups
- Rank samples by confidence scores per class
- Generate pairwise comparisons (e.g., Class A vs Class B)

---

## 2. Methods

### 2.1 Data Collection & Preparation

#### Input Data Summary

| Aspect | Details |
|--------|---------|
| **Annotation File** | [FILE NAME] |
| **Protein Matrix** | [FILE NAME] |
| **Total Samples (Raw)** | [N] samples |
| **Total Features (Raw)** | [N] proteins |
| **Missing Data** | [PERCENT]% values missing |

#### Class Distribution (Raw Data)

```
Class Distribution Before Filtering:
[CLASS1]:  [N_RAW] samples ([PCT]%)
[CLASS2]:  [N_RAW] samples ([PCT]%)
[CLASS3]:  [N_RAW] samples ([PCT]%)
[CLASS4]:  [N_RAW] samples ([PCT]%)
────────────────────────────
Total:     [N_TOTAL] samples

Imbalance Ratio: [RATIO]:[1] (largest:smallest)
```

#### Preprocessing Pipeline

```
Raw Data ([N_RAW] samples, [N_RAW_FEAT] features)
    ↓
1. Remove NA in annotation column
    → Removed [N] samples
    ↓
2. Keep only specified classes
    → Kept [N_CLASSES] classes, removed [N] samples
    ↓
3. Convert to numeric format
    → Failed [N] features, kept [N_FEAT] features
    ↓
4. Remove sparse features (>80% NA)
    → Removed [N] features
    ↓
Final Data ([N_FINAL] samples, [N_FINAL_FEAT] features)
```

#### Final Dataset Characteristics

| Class | N_Train | N_Test | Total | Percentage |
|-------|---------|--------|-------|-----------|
| [CLASS1] | [N_TR] | [N_TE] | [N_TOT] | [PCT]% |
| [CLASS2] | [N_TR] | [N_TE] | [N_TOT] | [PCT]% |
| [CLASS3] | [N_TR] | [N_TE] | [N_TOT] | [PCT]% |
| [CLASS4] | [N_TR] | [N_TE] | [N_TOT] | [PCT]% |
| **Total** | **[N]** | **[N]** | **[N]** | **100%** |

**Balance Assessment:**
```
Ideal:     All classes equal
Your Data: [ASSESSMENT]
  - Ratio of largest:smallest = [RATIO]:1
  - [ASSESSMENT: balanced/slightly imbalanced/highly imbalanced]
  - Impact: [MITIGATION_STRATEGY]
```

### 2.2 Machine Learning Architecture

#### Algorithm: XGBoost with Objective Function
```
Multiclass Strategy: [STRATEGY_TYPE]

Option A: Softmax (Multinomial) Objective
  ├─ Directly optimizes for all classes simultaneously
  ├─ Predicts probability for each class (sums to 1)
  ├─ Single model with internal one-vs-rest classification
  └─ Faster predictions

Option B: One-vs-Rest (OVR) Decomposition
  ├─ N separate binary models (one per class)
  ├─ Model 1: Class A vs (all other classes)
  ├─ Model 2: Class B vs (all other classes)
  ├─ Model 3: Class C vs (all other classes)
  └─ Allows individual ROC analysis per model

Option C: One-vs-One (OVO) Decomposition
  ├─ N×(N-1)/2 pairwise binary models
  ├─ Model 1: Class A vs Class B (2 classes)
  ├─ Model 2: Class A vs Class C (2 classes)
  ├─ Model 3: Class B vs Class C (2 classes)
  └─ [N_PAIRS] total pairwise models
```

**Your Implementation:** [STRATEGY_SELECTED]

#### Hyperparameter Optimization

**Method:** Bayesian Optimization with Gaussian Processes
- **Objective:** Maximize CV accuracy (macro-average across classes)
- **CV Strategy:** Stratified [K]-fold cross-validation (maintains class distribution)
- **Search Budget:** [N_ITER] iterations ([INIT_POINTS] random + [N_ITER-INIT_POINTS] Bayesian)
- **Acquisition Function:** Upper Confidence Bound (UCB)

#### Optimized Hyperparameters

| Parameter | Value | Ranges Searched | Explanation |
|-----------|-------|-----------------|-------------|
| **eta** | [VALUE] | [MIN]-[MAX] | Learning rate (lower=safer) |
| **max_depth** | [VALUE] | [MIN]-[MAX] | Tree complexity |
| **subsample** | [VALUE] | [MIN]-[MAX] | Row sampling (prevents overfitting) |
| **colsample_bytree** | [VALUE] | [MIN]-[MAX] | Feature sampling per tree |
| **min_child_weight** | [VALUE] | [MIN]-[MAX] | Minimum leaf samples |
| **gamma** | [VALUE] | [MIN]-[MAX] | Minimum split gain |
| **alpha** (L1) | [VALUE] | [MIN]-[MAX] | L1 regularization |
| **lambda** (L2) | [VALUE] | [MIN]-[MAX] | L2 regularization |

#### Fixed Training Parameters
```
booster:                   gbtree (gradient boosted trees)
objective:                 multi:softmax OR multi:softprob
num_class:                 [N_CLASSES]
eval_metric:               mlogloss OR auc (OVR/OVO)
nrounds:                   2000 max boosting iterations
early_stopping_rounds:     150 (stop if no improvement)
verbose:                   10 (report every 10 iterations)
random_state:              42 (reproducibility)
```

---

## 3. Results & Model Performance

### 3.1 Overall Classification Performance

#### Multi-Class Confusion Matrix (Test Set)

```
                    Predicted →
Actual ↓         [CLASS1]  [CLASS2]  [CLASS3]  [CLASS4]  Total
─────────────────────────────────────────────────────────────
[CLASS1]           [N]       [N]       [N]       [N]    [N]
[CLASS2]           [N]       [N]       [N]       [N]    [N]
[CLASS3]           [N]       [N]       [N]       [N]    [N]
[CLASS4]           [N]       [N]       [N]       [N]    [N]
─────────────────────────────────────────────────────────────
Total              [N]       [N]       [N]       [N]    [N]
```

**Diagonal (correct predictions):** [N] samples correct
**Off-diagonal (misclassifications):** [N] samples incorrect

#### Overall Classification Metrics

| Metric | Value | Interpretation |
|--------|-------|---|
| **Overall Accuracy** | [VALUE]% | Fraction of all predictions correct |
| **Macro F1-Score** | [VALUE] | Unweighted average of per-class F1 (0-1) |
| **Weighted F1-Score** | [VALUE] | Weighted by class size (more realistic) |
| **Matthews Correlation Coeff** | [VALUE] | Correlation for multiclass (-1 to 1) |

#### Class-Specific Performance

| Class | N_Actual | Accuracy | Precision | Recall | F1-Score |
|-------|----------|----------|-----------|--------|----------|
| [CLASS1] | [N] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS2] | [N] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS3] | [N] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS4] | [N] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| **Macro Avg** | **[N]** | **[VALUE]%** | **[VALUE]%** | **[VALUE]%** | **[VALUE]** |
| **Weighted Avg** | **[N]** | **[VALUE]%** | **[VALUE]%** | **[VALUE]%** | **[VALUE]** |

**Understanding Class-Specific Metrics:**
- **Accuracy (per-class):** How often this class is correctly identified
- **Precision:** Of predicted instances, how many actually were this class
- **Recall (Sensitivity):** Of actual instances, how many were correctly identified
- **F1-Score:** Harmonic mean of precision and recall

### 3.2 One-vs-One (OVO) Pairwise Analysis

**[N_PAIRS] Pairwise Comparisons (Each OVO Model):**

#### Pairwise Comparison Matrix

| Class Pair | AUC | Accuracy | Sensitivity | Specificity | Best_Threshold |
|-----------|-----|----------|-------------|-------------|-----------------|
| [CLASS1] vs [CLASS2] | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS1] vs [CLASS3] | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS1] vs [CLASS4] | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS2] vs [CLASS3] | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS2] vs [CLASS4] | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |
| [CLASS3] vs [CLASS4] | [VALUE] | [VALUE]% | [VALUE]% | [VALUE]% | [VALUE] |

**Interpretation of Pairwise Results:**
- **Highest AUC pairs:** Classes are most distinguishable (easy to separate)
- **Lowest AUC pairs:** Classes are most similar (difficult to separate, possible biological overlap)
- **All AUC > 0.70:** Good pairwise discrimination across all combinations

### 3.3 One-vs-Rest (OVR) Performance

**Alternative View: Each Class vs All Others**

| Class | AUC | Sensitivity | Specificity | Interpretation |
|-------|-----|-------------|-------------|-----------------|
| [CLASS1] vs Others | [VALUE] | [VALUE]% | [VALUE]% | Model at identifying [CLASS1] |
| [CLASS2] vs Others | [VALUE] | [VALUE]% | [VALUE]% | Model at identifying [CLASS2] |
| [CLASS3] vs Others | [VALUE] | [VALUE]% | [VALUE]% | Model at identifying [CLASS3] |
| [CLASS4] vs Others | [VALUE] | [VALUE]% | [VALUE]% | Model at identifying [CLASS4] |

---

## 4. Visualizations & Interpretability

### 4.1 ROC Curves

**Multiclass ROC Interpretation Options:**

**Option A: Multiple OVO Curves (Binary Comparisons)**
```
Locations: roc_curves_ovo_*.png (one per pairwise comparison)

Each plot shows:
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Blue curve: Pairwise discriminative ability
- Red dot: Operating point at optimal threshold
- AUC value: Discriminative power (0.50=random, 1.0=perfect)

Interpretation:
├─ Curve in top-left: Classes are very distinguishable
├─ Curve near diagonal: Classes are similar/overlapping
└─ High AUC (>0.7): Good separation for this pair
```

**Option B: One-vs-Rest Curves**
```
Locations: roc_curves_ovr_*.png (one per class)

Visual Show:
- How well the model identifies each class from all others
- Useful for understanding which classes are "easy" vs "hard"
```

### 4.2 Sample Predictions & Confidence

**Location:** `predicted_samples_[TIMESTAMP].png` (if multiclass visualization available)

```
What This Shows:
- Each sample's predicted probabilities for each class
- Color-coded by actual class membership
- Confidence in predictions (how separated are the predictions)
- Ranking by highest confidence prediction

Interpretation:
- Clear clustering: Model is confident
- Overlapping groups: Model is uncertain
- Within-class consistency: Good generalization
```

### 4.3 Predicted Probability Distributions

**Example Output Table:**

| Sample_ID | Class_Actual | Prob_[C1] | Prob_[C2] | Prob_[C3] | Prob_[C4] | Predicted | Confidence | Correct? |
|-----------|---|-----------|-----------|-----------|-----------|-----------|-----------|----------|
| [ID1] | [C1] | 0.92 | 0.05 | 0.02 | 0.01 | C1 | 92% | ✓ |
| [ID2] | [C1] | 0.78 | 0.15 | 0.05 | 0.02 | C1 | 78% | ✓ |
| [ID3] | [C2] | 0.10 | 0.88 | 0.01 | 0.01 | C2 | 88% | ✓ |
| [ID4] | [C3] | 0.20 | 0.30 | 0.35 | 0.15 | C3 | 35% | ⚠ Uncertain |
| [ID5] | [C4] | 0.40 | 0.20 | 0.25 | 0.15 | C1 | 40% | ✗ Incorrect |

**How to Read This:**
- **Prob columns:** Probability that sample belongs to each class (sum to 1.0)
- **Predicted:** Class with highest probability
- **Confidence:** The highest probability value (0.5 = maximum uncertainty, 1.0 = maximum certainty)
- **Correct?:** Whether predicted matches actual class

---

## 5. Feature Importance & Biological Insights

### 5.1 Top Predictive Proteins (Overall)

**Features Used by Model for Classification (Top 20):**

| Rank | Protein_ID | Importance | Associations |
|------|-----------|-----------|---------------|
| 1 | [PROT1] | [SCORE] | Differs between [CLASS1] and [CLASS2] |
| 2 | [PROT2] | [SCORE] | Elevated in [CLASS3], low in [CLASS4] |
| 3 | [PROT3] | [SCORE] | Complex multi-class pattern |
| ... | ... | ... | ... |
| 20 | [PROT20] | [SCORE] | Subtle differences |

### 5.2 Class-Specific Protein Signatures

**Top Proteins Associated with Each Class:**

#### [CLASS1] Signature Proteins
```
High in [CLASS1], Low in Others:
1. [PROTEIN]: 2.5-fold higher in [CLASS1]
2. [PROTEIN]: 1.8-fold higher in [CLASS1]
3. [PROTEIN]: Unique to [CLASS1]

Biological Pathway: [PATHWAY_IF_KNOWN]
Function: [FUNCTION]
```

#### [CLASS2] Signature Proteins
```
High in [CLASS2], Low in Others:
1. [PROTEIN]: 3.1-fold higher in [CLASS2]
2. [PROTEIN]: 2.2-fold higher in [CLASS2]
3. [PROTEIN]: Uniquely low in [CLASS2]

Biological Pathway: [PATHWAY_IF_KNOWN]
Function: [FUNCTION]
```

#### [CLASS3] Signature Proteins
```
High in [CLASS3], Low in Others:
1. [PROTEIN]: 1.6-fold higher in [CLASS3]
2. [PROTEIN]: Complex pattern

Biological Pathway: [PATHWAY_IF_KNOWN]
Function: [FUNCTION]
```

#### [CLASS4] Signature Proteins
```
High in [CLASS4], Low in Others:
1. [PROTEIN]: Expressed in [CLASS4]
2. [PROTEIN]: Variable pattern

Biological Pathway: [PATHWAY_IF_KNOWN]
Function: [FUNCTION]
```

### 5.3 Protein Interaction Patterns

**Key Observations:**
- Proteins [A], [B], [C] tend to be co-expressed and differentiate [CLASS1] vs [CLASS2]
- Proteins [X], [Y], [Z] separate [CLASS3] from others
- [CLASS4] shows partial overlap with [CLASS1] in protein [P]

---

## 6. Model Quality & Reliability Assessment

### 6.1 Cross-Validation Performance

**[K]-Fold Stratified Cross-Validation (Training Process):**

| Fold | Accuracy | Macro F1 | Weighted F1 | Notes |
|------|----------|----------|-------------|-------|
| Fold 1 | [VALUE]% | [VALUE] | [VALUE] | [NOTE] |
| Fold 2 | [VALUE]% | [VALUE] | [VALUE] | [NOTE] |
| Fold 3 | [VALUE]% | [VALUE] | [VALUE] | [NOTE] |
| Fold 4 | [VALUE]% | [VALUE] | [VALUE] | [NOTE] |
| Fold 5 | [VALUE]% | [VALUE] | [VALUE] | [NOTE] |
| **Mean ± SD** | **[VALUE]% ± [SD]%** | **[VALUE] ± [SD]** | **[VALUE] ± [SD]** | **Stability** |

**Consistency Check:**
```
Standard Deviation Interpretation:
├─ SD < 5%: ✓ Model is stable and reliable
├─ SD 5-10%: ⚠ Moderate variation between folds
└─ SD > 10%: ⚠ Model may be sensitive to dataset variation
```

### 6.2 Class-Specific Overfitting Analysis

| Class | Train_Accuracy | Test_Accuracy | Difference | Status |
|-------|----------------|---------------|-----------|--------|
| [CLASS1] | [VALUE]% | [VALUE]% | [VALUE]% | [✓/⚠] |
| [CLASS2] | [VALUE]% | [VALUE]% | [VALUE]% | [✓/⚠] |
| [CLASS3] | [VALUE]% | [VALUE]% | [VALUE]% | [✓/⚠] |
| [CLASS4] | [VALUE]% | [VALUE]% | [VALUE]% | [✓/⚠] |

**Overfitting Assessment:**
```
Train vs Test Difference:
├─ < 5%: ✓ Good generalization
├─ 5-10%: ⚠ Slight overfitting
└─ > 10%: ⚠ Significant overfitting (model memorized training data)
```

### 6.3 Dataset Size Adequacy

```
Sample Size Analysis:
├─ Current Training Samples: [N_TRAIN]
├─ Classes: [N_CLASSES]
├─ Features: [N_FEATURES]
├─ Minimum Recommended: max(100, 10 × features) = [MIN]
├─ Your Ratio: [RATIO] samples per feature
└─ Assessment: [ADEQUATE/MARGINAL/INSUFFICIENT]

Rule of Thumb:
- 10-100+ samples per class: Good
- <10 samples per class: May have sparse class coverage
- <50 total samples: High variance in estimates
```

### 6.4 Misclassification Analysis

**Most Common Errors:**

| True Class | Predicted as | Count | Prob | Likely Reason |
|-----------|--------------|-------|------|---|
| [CLASS1] | [CLASS2] | [N] | [PROB]% | Protein overlap between classes |
| [CLASS3] | [CLASS4] | [N] | [PROB]% | Biological similarity |
| [CLASS4] | [CLASS1] | [N] | [PROB]% | Outlier samples |

**Similar Class Pairs:**
```
Classes [A] and [B]:
- Highest error rate between them
- Protein profiles very similar
- Might consider combining or further investigation
```

---

## 7. Class-Specific Characteristics

### 7.1 Per-Class Sample Details

#### [CLASS1]
```
Sample Count:     [N_TRAIN] training, [N_TEST] test
Percentage:       [PCT]% of total
Characteristics:  [DESCRIPTION]
Key Features:     [TOP_3_PROTEINS]
Misclassified as: [CLASS_X]: [N] times, [CLASS_Y]: [N] times
Model Accuracy:   [VALUE]% (best among all pairs with [CLASS1])
```

#### [CLASS2]
```
Sample Count:     [N_TRAIN] training, [N_TEST] test
Percentage:       [PCT]% of total
Characteristics:  [DESCRIPTION]
Key Features:     [TOP_3_PROTEINS]
Misclassified as: [CLASS_X]: [N] times, [CLASS_Y]: [N] times
Model Accuracy:   [VALUE]%
```

#### [CLASS3]
```
Sample Count:     [N_TRAIN] training, [N_TEST] test
Percentage:       [PCT]% of total
Characteristics:  [DESCRIPTION]
Key Features:     [TOP_3_PROTEINS]
Misclassified as: [CLASS_X]: [N] times, [CLASS_Y]: [N] times
Model Accuracy:   [VALUE]%
```

#### [CLASS4]
```
Sample Count:     [N_TRAIN] training, [N_TEST] test
Percentage:       [PCT]% of total
Characteristics:  [DESCRIPTION]
Key Features:     [TOP_3_PROTEINS]
Misclassified as: [CLASS_X]: [N] times, [CLASS_Y]: [N] times
Model Accuracy:   [VALUE]%
```

### 7.2 Class Imbalance Handling

**Original Distribution:**
```
[CLASS1]: [N] samples ([PCT]%)
[CLASS2]: [N] samples ([PCT]%)
[CLASS3]: [N] samples ([PCT]%)
[CLASS4]: [N] samples ([PCT]%)
Imbalance Ratio: [RATIO]:1
```

**Balancing Strategy Applied:**
- [STRATEGY]: [DESCRIPTION]
  - Stratified split ensures all classes in train/test
  - Weighted loss function: provides higher weight to minority classes
  - Higher early_stopping_rounds: allows longer, more stable training

**Index of Imbalance:**
```
0 = perfect balance
< 0.2 = well balanced
0.2-0.5 = imbalanced
> 0.5 = highly imbalanced

Your Model: [VALUE] ([ASSESSMENT])
```

---

## 8. Clinical/Practical Implementation

### 8.1 Decision Rules

**For a New Sample with Predicted Probabilities:**

```
Step 1: Obtain protein expression profile
Step 2: Model outputs probability for each class:
        P([CLASS1]) = [VALUE]
        P([CLASS2]) = [VALUE]
        P([CLASS3]) = [VALUE]
        P([CLASS4]) = [VALUE]
        [All sum to 1.0]

Step 3: Decision logic:
        IF max(probability) > 0.70:
            Predicted Class = arg_max(probability)
            Confidence = HIGH
        ELSE IF max(probability) > 0.50:
            Predicted Class = arg_max(probability)
            Confidence = MODERATE ⚠ (review if critical)
        ELSE:
            UNCERTAIN - Request additional testing
            Confidence = LOW
            Recommend manual review
```

**Example Decision Context:**

```
Sample X shows:
- P(CLASS1) = 0.15
- P(CLASS2) = 0.65
- P(CLASS3) = 0.15
- P(CLASS4) = 0.05

Decision: CLASSIFY AS CLASS2 with 65% confidence
Action:   Moderate confidence - acceptable for screening
          But confirm with additional tests if critical decision
```

### 8.2 Cost-Benefit Analysis

**If Misclassifying [CLASS1] as [CLASS2] is expensive:**
```
Approach: Calibrate probabilities
- Lower acceptance threshold for [CLASS1] (>0.60 instead of 0.50)
- Use model probability outputs with caution in boundary cases
- Recommend secondary confirmation tests
```

**If Sensitivity to [CLASS3] is critical:**
```
Approach: Optimize for recall
- Focus on One-vs-Rest [CLASS3] model
- Lower probability threshold for [CLASS3] detection
- Accept higher false positive rate for more sensitivity
```

### 8.3 Deployment Considerations

**Recommended Implementation:**

1. **Primary Screening:** Use overall model with 0.70 confidence threshold
2. **Uncertain Cases (0.50-0.70 confidence):** Flag for review
3. **Critical Decisions:** Confirm with secondary method (e.g., targeted proteomics)
4. **Rare Classes:** [CLASS4] has lower accuracy [VALUE]% - use caution

**Quality Assurance:**
- Monthly performance monitoring on new samples
- Retrain if accuracy drops >5%
- Track misclassification patterns
- Document exceptions and learn from mistakes

---

## 9. Model Limitations & Important Caveats

### 9.1 Scope Limitations

⚠️ **What This Model CAN Do:**
✓ Classify new samples into one of [N_CLASSES] categories
✓ Provide probability estimates for each class
✓ Identify most important proteins for classification
✓ Achieve [ACCURACY]% accuracy on similar datasets

⚠️ **What This Model CANNOT Do:**
✗ Predict classes outside training classes (only: [CLASSES])
✗ Explain WHY proteins are different (association only)
✗ Work with different proteomics platforms without validation
✗ Handle systematic platform differences
✗ Predict continuous outcomes
✗ Extrapolate to new biological conditions

### 9.2 Data Dependencies

**Model requires:**
- Protein expression in numeric format
- [N_FEATURES] specific proteins measured
- Similar proteomics technology as training
- No systematic batch effects
- Similar population to training set

**Will fail if:**
- ✗ Proteins measured via different platform
- ✗ Dataset has missing [key proteins]
- ✗ Different tissue type than training
- ✗ Different disease stage/severity
- ✗ Completely different population (ethnicity, age, sex)

### 9.3 Known Weaknesses

| Weakness | Class Affected | Mitigation |
|----------|---|---|
| Lower accuracy | [CLASS]_X | Requires [N]+ more training samples |
| Confused with | [CLASS]_Y | Gets confused with similar biology |
| Sensitive to | [CONDITION] | Monitor this condition in deployment |
| Threshold | [VALUE] | May need adjustment for clinical use |

### 9.4 Class Confusion Patterns

**Classes Most Often Confused:**

```
[CLASS1] ↔ [CLASS2]:
├─ Error rate: [VALUE]%
├─ Reason: Both show high expression of protein [P]
├─ Suggestion: Could be biological subtypes of same class
└─ Recommendation: Consider merging or biological investigation

[CLASS3] ↔ [CLASS4]:
├─ Error rate: [VALUE]%
├─ Reason: Protein profiles highly similar
├─ Suggestion: These classes may be hard to distinguish
└─ Recommendation: Consult domain experts on clinical relevance
```

---

## 10. Reproducibility & Implementation

### 10.1 Generated Files

```
Output Directory: [OUTPUT_PATH]

Model & Parameters:
├── xgb_model_[TIMESTAMP].json          [SIZE] KB
│   └─ Trained multiclass XGBoost model
├── best_params_[TIMESTAMP].tsv         
│   └─ Optimized hyperparameters
├── training_log_[TIMESTAMP].log        
│   └─ Full Bayesian optimization history

Training Data:
├── train_matrix_[TIMESTAMP].tsv        [SIZE] MB
│   └─ Training samples × proteins
├── test_matrix_[TIMESTAMP].tsv         [SIZE] MB
│   └─ Test samples × proteins
├── feature_list_[TIMESTAMP].txt        
│   └─ [N_FEATURES] proteins used

Predictions & Results:
├── predicted_probabilities_[TIMESTAMP].tsv
│   └─ [N_TEST] samples × [N_CLASSES] probabilities
├── evaluation_results_[TIMESTAMP].tsv  
│   └─ Accuracy, per-class metrics
├── confusion_matrix_[TIMESTAMP].tsv    
│   └─ [N] × [N] classification matrix

Visualizations:
├── roc_curves_ovo_[TIMESTAMP]/         
│   ├─ roc_[CLASS1]_vs_[CLASS2].png
│   ├─ roc_[CLASS1]_vs_[CLASS3].png
│   ├─ roc_[CLASS1]_vs_[CLASS4].png
│   ├─ roc_[CLASS2]_vs_[CLASS3].png
│   ├─ roc_[CLASS2]_vs_[CLASS4].png
│   └─ roc_[CLASS3]_vs_[CLASS4].png
│
└── feature_importance_[TIMESTAMP].png  
    └─ Top [N] proteins by importance
```

### 10.2 Making Predictions on New Data

**Command:**
```bash
python py_scripts/cli.py apply \
  --model path/to/xgb_model_[TIMESTAMP].json \
  --protein new_protein_data.tsv \
  --annotation new_annotation.tsv \
  --annotcol class_column_name \
  --classes [CLASS1] [CLASS2] [CLASS3] [CLASS4] \
  --evaltsv path/to/evaluation_results_[TIMESTAMP].tsv \
  --output predictions_output_folder/
```

**Input File Formats:**

Annotation File (new_annotation.tsv):
```
sample_id    [COLUMN_NAME]
sample_1     [CLASS1]
sample_2     [CLASS2]
sample_3     [CLASS3]
sample_4     [CLASS4]
```

Protein File (new_protein_data.tsv):
```
             sample_1    sample_2    sample_3    sample_4
[PROTEIN_1]  0.456       0.123       NA          0.789
[PROTEIN_2]  0.234       0.567       0.345       0.678
[PROTEIN_3]  NA          0.432       0.567       0.234
...          ...         ...         ...         ...
```

**Output:** Probability predictions + visualizations for each sample

### 10.3 Retraining the Model

**When to Retrain:**
- Collected [N]+ new samples
- Cross-validation accuracy drops >5%
- Deployed model misses [N]+ samples
- New population or condition emerges
- Proteomics platform changes

**Retraining Command:**
```bash
python py_scripts/cli.py train \
  --annotation combined_annot.tsv \
  --protein combined_protein.tsv \
  --annotcol class_column \
  --classes [CLASS1] [CLASS2] [CLASS3] [CLASS4] \
  --output retrained_model/ \
  --n_iter 25 \
  --init_points 5 \
  --testsize 0.25 \
  --seed 42
```

---

## 11. Comparative Analysis

### 11.1 vs Binary Classification Models

```
Binary Model (e.g., [CLASS1] vs All Others):
├─ Simpler, faster training
├─ Individual AUC = [VALUE]
├─ Use when only need to identify one class
└─ Limitation: Cannot distinguish between other classes

Multiclass Model (This Model):
├─ More complex, thorough analysis
├─ Overall accuracy = [VALUE]%
├─ Use when all class distinctions matter
└─ Strength: Single model handles all classes
```

### 11.2 vs Traditional Methods

| Aspect | Traditional | XGBoost Multiclass |
|--------|-------------|---|
| **Handling of interactions** | Linear | Non-linear ✓ |
| **Feature selection** | Manual | Automatic ✓ |
| **Missing data** | Imputation-dependent | Robust ✓ |
| **Training speed** | Slow | Fast ✓ |
| **Feature importance** | Limited | Rich ✓ |
| **Interpretability** | Good | Moderate |

---

## 12. References & Technical Details

### 12.1 Key Publications

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123-140.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics.

### 12.2 Model Card

```
Model Name:             ProteoBoostR Multiclass Classifier
Version:                [VERSION]
Task:                   Multiclass classification (Proteomics)
Algorithm:              XGBoost with [STRATEGY]
Input Dimension:        [N_FEATURES] protein features
Output:                 [N_CLASSES] class probabilities
Training Samples:       [N_TRAIN]
Test Samples:           [N_TEST]
Test Accuracy:          [ACCURACY]%
Test Macro F1:          [F1]
Hyperparameter method:  Bayesian Optimization
Iterations:             [N_ITER]
Training time:          [TIME] minutes
```

### 12.3 Mathematical Details

**Softmax (Multinomial) Output:**
```
P(class = j | sample) = exp(f_j(X)) / Σ exp(f_k(X))  for all k

where:
- f_j(X) = XGBoost output for class j
- Probabilities sum to 1.0
- All values between 0 and 1
```

**One-vs-Rest Decomposition:**
```
For each class j:
  Binary model_j = class j vs (all others)
  Output: P(class j | sample) vs P(not class j | sample)

Final prediction = argmax_j(P(class j | sample))
```

**One-vs-One Pairwise:**
```
For each pair (i, j) where i < j:
  Binary model_{ij} = class i vs class j
  
Final vote = count votes for each class across all pairwise models
```

---

## 13. FAQ & Interpretation Guide

### Q1: What does a 0.75 probability mean?
**A:** The model is 75% confident the sample belongs to this class, 25% confidence it belongs to another. In practice:
- >0.70: Confident prediction (reasonable for most uses)
- 0.50-0.70: Moderate confidence (verify if critical)
- <0.50: Low confidence (don't trust, seek other data)

### Q2: Why was sample X misclassified?
**A:** Likely reasons:
- Sample has protein profile similar to multiple classes
- Actual sample was mislabeled in training data
- Outlier sample with unusual protein expression
- Classes share biological features

**Solution:** Review protein levels for this sample and compare to class signatures

### Q3: Can I use only the top 5 most important proteins?
**A:** Not recommended:
- Model uses interactions between many proteins
- Top 5 alone = ~[VALUE]% accuracy
- Full set = [VALUE]% accuracy
- Better to use all [N_FEATURES] proteins if possible

### Q4: The model performs worse on [CLASS]. Why?
**A:** Likely causes:
- Fewer training samples for this class ([N_SAMPLES])
- [CLASS] is biologically intermediate between others
- [CLASS] is less distinct in protein space
- Outliers in [CLASS] training data

**Solution:** Collect more [CLASS] samples, or merge with similar class

### Q5: How often should I update the model?
**A:** Retrain when:
- Collected [N]+ new samples (recommended: [N_MIN])
- Performance drops >5%
- Deployed >6 months
- New proteomics platform
- If confident in new biology, update quarterly

### Q6: Can this model predict probability of [CLASS] membership confidence?
**A:** Yes! The output probability IS the confidence measure. But note:
- Model was trained to maximize accuracy, not calibration
- Probabilities may be overconfident or underconfident
- For critical decisions, add secondary confirmation

### Q7: What happens with a completely new [CLASS5]?
**A:** The model CANNOT predict a class it wasn't trained on. Instead:
- Will force-assign to one of [CLASS1-4]
- Probability distribution will be scattered
- DO NOT trust this prediction
- Collect [CLASS5] samples and retrain if needed

### Q8: Why is class [X] hardest to distinguish?
**A:** Analysis shows:
- [CLASS_X] protein profile overlaps most with [CLASS_Y]
- Only [N] discriminative proteins
- Pairwise AUC with [CLASS_Y] is [VALUE]
- Consider if classes should be merged or separated biologically

---

## 14. Approval & Sign-Off

| Role | Name | Signature | Date | Notes |
|------|------|-----------|------|-------|
| Model Developer | [NAME] | __________ | __________ | Built model |
| Bioinformatician | [NAME] | __________ | __________ | Validated methods |
| Domain Expert | [NAME] | __________ | __________ | Reviewed biology |
| QA/Reviewer | [NAME] | __________ | __________ | Quality check |
| Approver | [NAME] | __________ | __________ | Final approval |

---

## Appendix A: Detailed Evaluation Metrics

**Multiclass Contingency Table:**
```
[Detailed table omitted for space, but included in actual report]
```

**Per-Class Metrics (Detailed):**
```
For each class, including:
- True Positive Rate (Sensitivity)
- False Positive Rate (1-Specificity)
- Positive Predictive Value (Precision)
- Negative Predictive Value
- F1-Score
- Matthews Correlation
```

---

## Appendix B: Sample Case Studies

**Sample [ID1]: Clear [CLASS1]**
```
Proteins:  [P1]=0.95, [P2]=0.88, [P3]=0.05
Prediction: [CLASS1] with 98% confidence
Reason: All signature proteins for [CLASS1] present
Status: ✓ High confidence, expected result
```

**Sample [ID2]: Ambiguous [CLASS3]**
```
Proteins:  [P1]=0.45, [P2]=0.52, [P3]=0.48
Prediction: [CLASS3] with 52% confidence
Reason: Borderline protein levels across classes
Status: ⚠ Low confidence, recommend manual review
```

---

**Report Generated:** [DATE TIME]  
**By:** [ANALYST NAME]  
**Institution:** [ORG]  
**Version:** 1.0  
**Contact:** [EMAIL]
