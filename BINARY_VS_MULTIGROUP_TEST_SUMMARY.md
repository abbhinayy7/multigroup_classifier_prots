# Binary vs Multigroup Classification Test - Summary

## Test Overview

This comprehensive test compares ProteoBoostR's performance using two approaches:
1. **Binary Classification** (2 classes)
2. **Multigroup Classification** (3+ classes)

## Datasets Used

### Binary Model: GBM Werner Dataset
- **Total Samples**: 55 samples
- **Proteins (Features)**: 9,731
- **Classification Task**: Subtype1 vs Subtype2+3+4
- **Class Distribution**: 11 subtype1 vs 44 others (imbalanced)
- **Train/Test Split**: 41 training, 14 test

### Multigroup Model: Multigroup Test Data
- **Total Samples**: 92 samples  
- **Proteins (Features)**: 10,718
- **Classification Task**: 3-class (Control, GroupA, GroupB)
- **Class Distribution**: 
  - Control: 26 (28%)
  - GroupA: 29 (32%)
  - GroupB: 37 (40%)
- **Train/Test Split**: 69 training, 23 test

## Methodology

### Hyperparameter Optimization
Both models use **Bayesian Optimization** with the same hyperparameter search space:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| eta (learning rate) | 0.01 - 0.30 | Controls step size of boosting |
| max_depth | 1 - 10 | Maximum tree depth |
| subsample | 0.5 - 1.0 | Row sampling for each tree |
| colsample_bytree | 0.5 - 1.0 | Column sampling for each tree |
| min_child_weight | 1 - 10 | Minimum weight in child node |
| gamma | 0 - 5 | Minimum loss reduction |
| alpha (L1) | 0 - 10 | L1 regularization |
| lambda (L2) | 0 - 10 | L2 regularization |

**Optimization Strategy**: 
- 8 random initialization points
- 5 Bayesian optimization iterations
- 5-fold cross-validation for each parameter set
- Total evaluations: 13 per model

### Model Training
- **Algorithm**: XGBoost
- **Early Stopping**: 100 rounds without improvement
- **Maximum Boosting Rounds**: 1,000
- **Binary Objective**: binary:logistic with AUC metric
- **Multiclass Objective**: multi:softmax with mlogloss metric

## Expected Results

### Metrics Captured

**For Binary Classification**:
- Accuracy
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)  
- Precision
- F1-Score
- AUC-ROC

**For Multigroup Classification**:
- Overall Accuracy
- Precision (weighted average across classes)
- Recall (weighted average across classes)
- F1-Score (weighted average across classes)
- Per-class performance breakdown

### Performance Comparison
The test will provide:
1. **Side-by-side metrics comparison**
2. **Model complexity analysis** (decision boundaries, boosting rounds)
3. **Use case recommendations**
4. **When to use each approach**

## Key Insights from Comparison

### Binary Classification Strengths
- Simpler decision boundary (single threshold)
- Higher interpretability for yes/no decisions
- Better AUC/ROC analysis possible
- Optimal for dichotomous problems

### Multigroup Classification Strengths
- Handles multiple distinct categories in one model
- No need for separate binary vs binary comparisons
- Can handle class imbalance across multiple groups
- More biologically relevant for disease subtypes

## Test Duration
Expected runtime: 10-15 minutes (Bayesian optimization is computationally intensive)

## Output Files
- `test_results.txt` - Complete test output with all metrics
- This summary document

## Running the Test

```bash
cd g:\ProteoBoostR
python test_binary_vs_multigroup.py
```

## Interpreting Results

### When Binary is Better
- Lower test error rates suggest binary is more suited to the data
- If one class dominates, binary can achieve higher accuracy

### When Multigroup is Better
- If multigroup accuracy is competitive, it's more versatile
- Useful when you want to distinguish multiple disease states
- Can reduce to binary by grouping similar classes later

### Optimal Strategies
- Use **Binary** for population screening (disease yes/no)
- Use **Multigroup** for disease typing and subtyping
- Use **Binary Cascade** for hierarchical classification (first binary, then binary on positive class)

---

**Test Script**: `test_binary_vs_multigroup.py`
**Completion Status**: Running...
