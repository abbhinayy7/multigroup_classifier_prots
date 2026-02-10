# How to Complete Your ProteoBoostR Model Reports
## Step-by-Step Guide for Binary & Multigroup Models

---

## Overview

You now have **two comprehensive report templates**:
1. **BINARY_MODEL_REPORT_TEMPLATE.md** - For 2-class models (e.g., Cancer vs Normal)
2. **MULTIGROUP_MODEL_REPORT_TEMPLATE.md** - For 3+ class models (e.g., Subtype A vs B vs C)

This guide shows you **exactly where to find each piece of information** to fill in the [BRACKETED] placeholders.

---

## Part 1: Gathering Information from Your Model

### Step 0: Run Your Model & Collect All Output Files

```bash
# For Binary Classification
python py_scripts/cli.py train \
  --annotation your_annot.tsv \
  --protein your_protein.tsv \
  --annotcol class_column \
  --neg negative_label \
  --pos positive_label \
  --output results_folder/

# For Multiclass Classification
python py_scripts/cli.py train \
  --annotation your_annot.tsv \
  --protein your_protein.tsv \
  --annotcol class_column \
  --output results_folder/
```

**Save the output folder** - You'll extract all report values from these files.

---

## Part 2: Extracting Values from Output Files

### **1. Model Performance Metrics**

**Find in:** `evaluation_results_[TIMESTAMP].tsv`

Open this file in Excel or text editor. It should look like:

```
Metric                    Value
Accuracy                  0.9818
Sensitivity               0.9773
Specificity               1.0000
Precision                 1.0000
F1_Score                  0.9885
AUC                       1.0000
Best_Threshold            0.7614
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Executive Summary: `[ACCURACY]` | 98.18% | Accuracy × 100 |
| Executive Summary: `[AUC_VALUE]` | 1.0000 | AUC column |
| Section 3.1: `[VALUE]` fields | See above | Individual metric rows |
| Section 5: `[THRESHOLD]` | 0.7614 | Best_Threshold |

---

### **2. Confusion Matrix Data**

**Find in:** `confusion_matrix_[TIMESTAMP].tsv`

```
         Predicted_Negative  Predicted_Positive
Negative              10                      0
Positive               0                      7
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 3.1: Confusion Matrix | Fill grid | Use exact values |
| Section 3.1: `[TP]` | 7 | Predicted_Positive, Positive row |
| Section 3.1: `[TN]` | 10 | Predicted_Negative, Negative row |
| Section 3.1: `[FP]` | 0 | Predicted_Negative, Positive row |
| Section 3.1: `[FN]` | 0 | Predicted_Positive, Negative row |

---

### **3. Predicted Probabilities & Sample Predictions**

**Find in:** `predicted_probabilities_[TIMESTAMP].tsv`

```
sample_id         actual_label    predicted_prob    rank
sample_1          0               0.15              17
sample_2          1               0.92              1
sample_3          0               0.08              18
...
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 3.3: Predictions Table | Use top/bottom rows | Sample from this file |
| Section 4.2: `[N_TEST]` | Count of rows | Number of samples |
| Section 3.3: Confidence examples | Use actual values | predicted_prob column |

---

### **4. Best Hyperparameters**

**Find in:** `best_params_[TIMESTAMP].tsv`

```
Parameter    Value
eta          0.3576
max_depth    3
subsample    0.9366
colsample    0.7834
min_child    4
gamma        1.2345
alpha        0.5678
lambda       2.3456
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 2.2: Optimized Hyperparameters table | Fill all rows | Parameter and Value columns |
| Section 5: `[VALUE]` in each row | See above | Copy directly |

---

### **5. Training Datasets**

**Find in:** `train_matrix_[TIMESTAMP].tsv`

```
Check: 
- Number of rows = N_TRAIN (excluding header)
- Number of columns = N_FEATURES + 1 (sample_id + proteins)
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 2.1: `[N_TRAIN]` | Row count | Count all data rows |
| Section 2.1: `[N_FEATURES]` | Col count - 1 | Subtract 1 for sample_id |
| Section 2.1: Features description | "XXX proteins" | Use column count |

---

### **6. Test Datasets**

**Find in:** `test_matrix_[TIMESTAMP].tsv`

```
Same structure as train_matrix
- Number of rows = N_TEST
- Number of columns = same proteins as training
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 2.1: `[N_TEST]` | Row count | Count all data rows |
| Section 3.1: Test set reference | [N_TEST] samples | Use this count |

---

### **7. Class Distribution**

**Find in:** Your original annotation file or `train_matrix_[TIMESTAMP].tsv` (last column)

Count samples by class:

```bash
# Quick command (PowerShell):
$data = Import-Csv "train_matrix_[TIMESTAMP].tsv" -Delimiter "`t"
$data | Group-Object -Property "class_column" | Select-Object Name, Count
```

**Output might be:**
```
Name   Count
0      38
1      17
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 2.1: Class Distribution | Table | Use counts above |
| Section 5: `[N_TRAINING]` pos/neg | 17/38 | From counts |
| Executive Summary: Distribution | "17 pos, 38 neg" | Use these counts |

---

## Part 3: Adding Contextual Information

### **What You Need to Fill In Manually**

These require **your knowledge** of the data and biology:

| Section | What to Fill | Example |
|---|---|---|
| **[DATASET NAME]** | Your dataset name | "GBM Tumor Classification" |
| **[DATE]** | Report date | "February 10, 2026" |
| **[POSITIVE CLASS]** | What does class 1 represent? | "Aggressive Subtype" |
| **[NEGATIVE CLASS]** | What does class 0 represent? | "Standard Subtype" |
| **[DESCRIPTION]** | Why these classes matter | "Aggressive subtype requires different treatment" |
| **[USE CASE 1-3]** | How will this be used? | "Guide treatment selection", "Identify high-risk patients" |
| **[INTERPRETATION]** | What do the proteins mean? | "Immunological markers", "Cell cycle regulators" |

---

## Part 4: Extracting Protein Importance

### **Find Feature Importance**

The model should generate feature importance (if code includes it). If not, you can extract from model:

**Check output folder for:**
- `feature_importance_[TIMESTAMP].png` (visualization)
- Or check xgboost model directly

**Manual Extraction (Python):**
```python
import xgboost as xgb
import pandas as pd

# Load model
model = xgb.Booster()
model.load_model("xgb_model_[TIMESTAMP].json")

# Get feature importance
importance = model.get_score(importance_type='weight')
importance_df = pd.DataFrame(sorted(importance.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True), 
                           columns=['Feature', 'Importance'])

# Top 10
print(importance_df.head(10))
print(importance_df.to_csv("feature_importance.tsv", sep='\t', index=False))
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 5.1: Top 10 proteins | Table | feature_importance output |
| Section 5.1: `[PROTEIN]` names | Feature column | Model importance |
| Section 5.1: `[SCORE]` | Importance score | Importance column |

---

## Part 5: Cross-Validation Results

### **Find CV Performance**

**Check in:** Training log file or terminal output

In your `proteoboostr_[TIMESTAMP].log`:

```
[2026-01-28 15:03:51] Fold 1/5 - AUC: 0.9833, Accuracy: 0.9741
[2026-01-28 15:04:12] Fold 2/5 - AUC: 0.9950, Accuracy: 0.9825
[2026-01-28 15:04:33] Fold 3/5 - AUC: 0.9917, Accuracy: 0.9781
[2026-01-28 15:04:54] Fold 4/5 - AUC: 0.9875, Accuracy: 0.9703
[2026-01-28 15:05:15] Fold 5/5 - AUC: 0.9958, Accuracy: 0.9846
```

**Calculate statistics:**
```
Mean AUC:        (0.9833 + 0.9950 + 0.9917 + 0.9875 + 0.9958) / 5 = 0.9907
SD AUC:          Calculate using Excel =STDEV() function
Mean Accuracy:   (97.41 + 98.25 + 97.81 + 97.03 + 98.46) / 5 = 97.79%
SD Accuracy:     Calculate using Excel
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 6.1: CV Results table | Fold-by-fold values | Log file |
| Section 6.1: Mean ± SD | Calculated values | Excel calculation |

---

## Part 6: ROC Curve Information

### **Find in:** `roc_curve_20260128165741.png`

You'll need to visually inspect the ROC curve plot to extract:

```
Visual Elements:
- Look at x-axis: Goes from 0 (left) to 1 (right) = FPR
- Look at y-axis: Goes from 0 (bottom) to 1 (top) = TPR
- Find red dot: This is your operating point
- Read the AUC value from legend: "AUC = 0.9999"

For your report:
├─ AUC value: Already have from evaluation_results.tsv
├─ Threshold: Already have (0.7614)
├─ Curve shape: Describe as "top-left" / "diagonal" / "perfect"
└─ Operating point: Visual position on the curve
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 3.2: ROC interpretation | Describe shape | Visual inspection |
| Section 3.2: `[THRESHOLD]` | 0.7614 | evaluation_results.tsv |
| Section 3.2: Operating point description | "Near perfect" | Visual assessment |

---

## Part 7: For MULTIGROUP Models Only

### **Extra Data for Multiclass**

If using multigroup model, also extract:

**From evaluation results:**
```
Overall Accuracy: 88.2% (correct across all classes)
Macro F1-Score: 0.8634
Weighted F1-Score: 0.8742
Per-Class Accuracy:
  Class1: 92.1%
  Class2: 85.3%
  Class3: 88.2%
  Class4: 81.4%
```

**Pairwise Comparisons (if OVO analysis done):**
```
OVO ROC Results:
Class1 vs Class2: AUC = 0.965
Class1 vs Class3: AUC = 0.942
Class1 vs Class4: AUC = 0.928
Class2 vs Class3: AUC = 0.887
Class2 vs Class4: AUC = 0.856
Class3 vs Class4: AUC = 0.912
```

**Where these go in your report:**

| Template Section | Value | From File |
|---|---|---|
| Section 3.1: Overall metrics | All values | evaluation_results.tsv |
| Section 3.2: Pairwise table | All OVO AUC values | OVO analysis output |
| Section 5.2: Per-class accuracy | Individual values | Per-class section of results |

---

## Part 8: Filling in Quality Assessment Sections

### **Overfitting Analysis**

Compare train vs test accuracy:

```bash
python -c "
train_acc = 98.82  # From CV on training data
test_acc = 98.18   # From evaluation_results
diff = train_acc - test_acc
print(f'Train: {train_acc}%, Test: {test_acc}%, Diff: {diff}%')
"
```

**Assessment:**
- If diff < 5%: Write "✓ Good generalization - minimal overfitting"
- If diff 5-10%: Write "⚠ Slight overfitting detected"
- If diff > 10%: Write "⚠ Significant overfitting - model learned training data"

**Where this goes in your report:**

| Template Section | Value |
|---|---|
| Section 6.1: Overfitting assessment | Use formula above |
| Section 6.1: Interpretation | Write assessment |

---

## Part 9: Quick Reference - What Goes Where

### **Universal Fields (All Reports)**

```
HEADER:
[DATE] = Current date                  → Use today's date
[DATASET NAME] = Your dataset          → E.g., "GBM_Werner"
[MODEL TIMESTAMP] = From filename      → E.g., "20260128165351"

PERFORMANCE SECTION:
[ACCURACY]% = accuracy × 100           → From evaluation_results
[AUC_VALUE] = AUC                      → From evaluation_results
[THRESHOLD] = Best_Threshold           → From evaluation_results
[N_TRAINING] = Training samples        → From train_matrix row count
[N_TEST] = Test samples                → From test_matrix row count
[N_FEATURES] = Features                → Column count - 1
```

### **Binary-Specific Fields**

```
[POSITIVE CLASS] = Class 1 name        → E.g., "Aggressive"
[NEGATIVE CLASS] = Class 0 name        → E.g., "Standard"
[TP] = True positives                  → From confusion_matrix
[TN] = True negatives                  → From confusion_matrix
[FP] = False positives                 → From confusion_matrix
[FN] = False negatives                 → From confusion_matrix
```

### **Multigroup-Specific Fields**

```
[N_CLASSES] = Number of classes        → E.g., 4
[CLASS1], [CLASS2], etc. = Names       → E.g., "Subtype_A", "Subtype_B"
[N_PAIRS] = Number of pairs            → For 4 classes: 4×3/2 = 6
[AUC_MACRO] = Macro-average AUC       → Average of all pairwise AUCs
```

---

## Part 10: Tips for Professional Presentation

### **Make It Look Great**

1. **Use consistent formatting:**
   - Numbers: "98.18%" not "0.9818"
   - Decimals: 4 places for probabilities, 2% for percentages
   - Tables: Align columns, use consistent fonts

2. **Add context:**
   - Don't just say "95% accuracy" - explain what that means
   - Compare to baseline: "vs 50% random guessing"
   - Explain implications: "This level of accuracy is suitable for..."

3. **Use visuals:**
   - Include the ROC curve PNG image
   - Include sample prediction plot PNG image
   - Consider adding confusion matrix heatmap

4. **For your audience:**
   - **Non-technical:** Emphasize practical meaning, minimize technical jargon
   - **Technical:** Include detailed metrics, thresholds, statistical details
   - **Mixed:** Provide both summaries and technical appendices

---

## Part 11: Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| Confusing accuracy with AUC | AUC is more robust to class imbalance |
| Forgetting units (%, decimal) | Always specify: "0.9818" vs "98.18%" |
| Using train accuracy as final metric | ALWAYS use test accuracy |
| Missing field `[DATE]` | Update date field before sharing |
| Ignoring class imbalance | Note if classes are unbalanced |
| Not explaining what each metric means | Add definition in parentheses |
| Copy-pasting template structure with placeholders | Search & Replace all `[]` before sharing |

---

## Part 12: Validation Checklist

Before sharing your report, verify:

- [ ] All `[BRACKETED]` fields are filled in with real values
- [ ] Numbers match between different sections (no inconsistencies)
- [ ] Date is current
- [ ] Dataset name is clear
- [ ] Class/group names are consistent throughout
- [ ] Performance metrics make sense (e.g., Sensitivity ≤ 100%)
- [ ] Threshold value is between 0 and 1
- [ ] Sample numbers match (N_TRAIN + N_TEST = total)
- [ ] Confusion matrix sums correctly
- [ ] ROC image is included and visible
- [ ] Spell-check completed
- [ ] Report is saved with descriptive filename

---

## Part 13: Filing & Organization

### **Recommended File Structure**

```
PROJECT_FOLDER/
├── [ProjectName]_Binary_Model_Report_Feb2026.md
├──[ProjectName]_Multigroup_Model_Report_Feb2026.md
├── model_outputs/
│   ├── xgb_model_20260128165351.json
│   ├── evaluation_results_20260128165741.tsv
│   ├── best_params_20260128165351.tsv
│   ├── confusion_matrix_20260128165741.tsv
│   ├── roc_curve_20260128165741.png
│   ├── predicted_samples_20260128165741.png
│   └── proteoboostr_20260128165351.log
└── [ProjectName]_Report_Guide.md (this file)
```

### **Naming Convention**

```
Good:
- GBM_Binary_Model_Report_Feb2026.md
- Proteomics_Multigroup_Report_v1.0.md
- Project_X_Clinical_Model_2026.md

Avoid:
- report.md (too vague)
- [TEMPLATE].md (will confuse with template)
- model (no extension)
```

---

## Part 14: Sharing & Presentation

### **Format for Different Audiences**

**For Executives:**
- Print just: Executive Summary + Key Results + Visualizations
- Use bullet points
- Avoid technical details

**For Domain Experts:**
- Full report including Methods section
- Include feature importance proteins  
- Discuss biological implications

**For ML/Data Scientists:**
- Include hyperparameters section
- Add CV details and regularization
- Include troubleshooting section

### **Export to PDF**

If using Markdown (.md file):
```bash
# Install pandoc if needed
pip install pandoc

# Convert to PDF
pandoc [FILENAME].md -o [FILENAME].pdf
```

---

## Summary

**You now have a complete system:**

1. ✅ **Template 1:** BINARY_MODEL_REPORT_TEMPLATE.md (for 2-class models)
2. ✅ **Template 2:** MULTIGROUP_MODEL_REPORT_TEMPLATE.md (for 3+ class models)
3. ✅ **This guide:** Step-by-step instructions on how to fill them in

**Next steps:**

1. Run your model and collect output files
2. Extract values using the guides above
3. Fill in the report template
4. Add context about your specific dataset
5. Share with stakeholders

---

**Questions?** See the FAQ sections in each template for common interpretations.

**Last Updated:** February 10, 2026
