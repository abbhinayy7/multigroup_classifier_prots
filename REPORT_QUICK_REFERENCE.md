# ProteoBoostR Model Report - Quick Reference Card

**PRINT THIS PAGE** and keep it handy while completing your report!

---

## 1️⃣ WHICH TEMPLATE TO USE?

| Your Model Has... | Use This Report |
|---|---|
| **2 classes** (e.g., Yes/No, Case/Control) | **BINARY_MODEL_REPORT_TEMPLATE.md** |
| **3+ classes** (e.g., Type A/B/C/D) | **MULTIGROUP_MODEL_REPORT_TEMPLATE.md** |

---

## 2️⃣ KEY VALUES & WHERE TO FIND THEM

Copy these from your model output files:

### From `evaluation_results_[TS].tsv`:
```
[ACCURACY]          = Accuracy value × 100 (add %)
[AUC_VALUE]         = AUC value (should be 0-1)
[THRESHOLD]         = Best_Threshold value
[SENSITIVITY]       = Sensitivity value × 100 (add %)
[SPECIFICITY]       = Specificity value × 100 (add %)
[PRECISION]         = Precision value × 100 (add %)
```

### From `confusion_matrix_[TS].tsv`:
```
[TP] = True Positives (correctly predicted positive)
[TN] = True Negatives (correctly predicted negative)
[FP] = False Positives (incorrectly predicted positive)
[FN] = False Negatives (incorrectly predicted negative)

Quick check: TP + TN + FP + FN = total test samples
```

### From `train_matrix_[TS].tsv` & `test_matrix_[TS].tsv`:
```
[N_TRAIN]  = Row count of train_matrix (# training samples)
[N_TEST]   = Row count of test_matrix (# test samples)
[N_FEATURES] = Column count - 1 (subtract sample_id column)
```

### From `best_params_[TS].tsv`:
```
[eta]      = learning rate
[max_depth] = tree depth parameter
[subsample] = subsample parameter
[colsample_bytree] = feature sampling
[min_child_weight] = leaf sample minimum
[gamma]    = split gain threshold
[alpha]    = L1 regularization
[lambda]   = L2 regularization
```

### From filename:
```
[TIMESTAMP] = Model creation date (from filename)
[DATE]      = Today's date when writing report
```

---

## 3️⃣ COMMON CALCULATIONS

**Calculate Standard Deviation (SD) from CV results:**
```
In Excel: =STDEV(range of fold accuracies)
Example: =STDEV(97.4%, 98.2%, 97.8%, 97.0%, 98.4%)
Result: ±0.65%
```

**Verify Confusion Matrix:**
```
Sensitivity = TP / (TP + FN)    [Higher = better at catching positives]
Specificity = TN / (TN + FP)    [Higher = better at confirming negatives]
Precision = TP / (TP + FP)      [Higher = fewer false alarms]
Accuracy = (TP + TN) / Total    [Overall correctness]
```

**Check if Balanced:**
```
Class 0 count: [N]
Class 1 count: [N]
Ratio: [larger] / [smaller]
Assessment:
  - <1.5:1 → Balanced ✓
  - 1.5-3:1 → Slightly imbalanced
  - >3:1 → Highly imbalanced
```

---

## 4️⃣ SECTION CHECKLIST

Fill these sections in this order:

```
┌─ HEADER & EXECUTIVE SUMMARY
│  ├─ [DATE]              → Today's date
│  ├─ [DATASET NAME]      → Your dataset name
│  ├─ [MODEL TIMESTAMP]   → From filename
│  ├─ [ACCURACY]%         → evaluation_results.tsv
│  ├─ [AUC_VALUE]         → evaluation_results.tsv
│  └─ [THRESHOLD]         → evaluation_results.tsv
│
├─ BACKGROUND SECTION
│  ├─ [POSITIVE CLASS]    → Your domain knowledge
│  ├─ [NEGATIVE CLASS]    → Your domain knowledge
│  └─ [USE CASES]         → Why you built this model
│
├─ METHODS SECTION
│  ├─ [N_SAMPLES]         → Annotation file
│  ├─ [N_FEATURES]        → Protein matrix columns - 1
│  ├─ [N_TRAIN]           → train_matrix row count
│  ├─ [N_TEST]            → test_matrix row count
│  └─ Hyperparameters     → best_params.tsv
│
├─ RESULTS SECTION
│  ├─ [TP], [TN], [FP], [FN]  → confusion_matrix.tsv
│  ├─ All percentages         → evaluation_results.tsv
│  ├─ ROC image               → roc_curve_[TS].png
│  └─ Feature importance      → top proteins list
│
├─ QUALITY SECTION
│  ├─ Train vs Test accuracy  → Compare CV vs test
│  ├─ CV results              → From training log
│  └─ Threshold analysis      → evaluation_results
│
└─ CONTEXT (Your Knowledge!)
   ├─ Why this classification matters
   ├─ What the top proteins do
   ├─ Biological interpretation
   ├─ Limitations
   └─ Next steps
```

---

## 5️⃣ QUICK INTERPRETATION GUIDE

### **Is My Accuracy Good?**
```
< 70%  → Poor (worse than many simple methods)
70-80% → Acceptable (reasonable discrimination)
80-90% → Good (strong separation)
> 90%  → Excellent (very predictive)
```

### **Is My AUC Good?**
```
< 0.60 → Poor discrimination
0.60-0.70 → Fair
0.70-0.80 → Good
0.80-0.90 → Very good
> 0.90 → Excellent
1.00 → Perfect (caution: possible overfitting)
```

### **Is There Overfitting?**
```
Train Acc - Test Acc = Difference

< 5%   → Good generalization ✓
5-10%  → Slight overfitting (⚠ acceptable)
> 10%  → Significant overfitting (⚠ concerning)
```

### **Sensitivity vs Specificity Trade-off**
```
High Sensitivity (>95%):
  ✓ Catches most positives (good for screening)
  ✗ More false positives (more follow-up tests)

High Specificity (>95%):
  ✓ Fewer false alarms (confident predictions)
  ✗ Misses some true positives (might miss cases)

Balanced (80-90%):
  ✓ Good overall performance
  → Best for most clinical use
```

---

## 6️⃣ RED FLAGS ⚠️

Stop and review if you see:

```
☐ Accuracy = 99% but dataset is only 20 samples
  → Likely overfitting, results may not generalize

☐ AUC = 0.5 (random guessing)
  → Model not working, check data/parameters

☐ Sensitivity = 100%, Specificity = 100%
  → Likely data contamination or leakage

☐ Feature importance very different from biology
  → Unexpected proteins, verify interpretation

☐ Test accuracy much worse than CV accuracy
  → Overfitting or different data distribution

☐ Sample numbers don't add up
  → Check: N_TRAIN + N_TEST = expected total

☐ [BRACKETED] placeholders still visible
  → You missed filling something in!
```

---

## 7️⃣ BEFORE YOU SHARE - VALIDATION CHECKLIST

```
Content Check:
☐ All [FILLED] with real values (no brackets left)
☐ Numbers consistent across sections
☐ Date field updated to today
☐ Dataset/model names consistent
☐ TP + TN + FP + FN = N_TEST samples

Math Check:
☐ Sensitivity = TP/(TP+FN), should equal report value
☐ Specificity = TN/(TN+FP), should equal report value
☐ Accuracy = (TP+TN)/Total, should equal report value
☐ Probabilities between 0-1
☐ Percentages between 0-100%

Format Check:
☐ No spelling errors
☐ Headings are consistent
☐ Tables aligned properly
☐ Images/plots visible
☐ File saved with clear name

Biology Check:
☐ Results make biological sense
☐ Top proteins are plausible
☐ Conclusions not overclaimed
☐ Limitations acknowledged
☐ Methods reproducible
```

---

## 8️⃣ TOP MISTAKES TO AVOID

| Mistake | Wrong | Right |
|---|---|---|
| **Confusing formats** | 0.9818 vs 98.18% | Specify which! Use 98.18% for percentages |
| **Wrong metric** | Using train accuracy as final | Always use test accuracy |
| **Missing reference** | "98% accuracy" | "98.18% accuracy on 17 test samples" |
| **Unit confusion** | "[THRESHOLD]" = 0.7614 | "[THRESHOLD]" = 0.7614 (already decimal, no %) |
| **Forgetting sample size** | "AUC = 0.95" (sounds great!) | "AUC = 0.95 on N=20 samples" (different perception) |
| **Math errors** | TP=7, FN=0, Sensitivity="100%" | ✓ Correct: 7/(7+0)=100% |
| **Unfilled placeholders** | Report has "[VALUE]" in it | Replace ALL [BRACKETS] before sharing |
| **Wrong template** | Binary model using multigroup template | Match template to number of classes |

---

## 9️⃣ FILE NAMING EXAMPLES

**GOOD Examples:**
```
✓ GBM_Binary_Model_Report_Feb2026.md
✓ Proteomics_Aggressive_vs_Standard_Report.md
✓ Project_Multigroup_Subtype_Classification_v1.md
✓ LUAD_4Class_Model_Report_2026-02-10.md
```

**AVOID:**
```
✗ report.md (too vague)
✗ [TEMPLATE].md (sounds like unfinished)
✗ model (missing extension)
✗ final_final_v3_realdone.md (unprofessional)
```

---

## 🔟 AFTER COMPLETION

```
Step 1: Save as .md file
  → File > Save As > [ProjectName]_Report.md

Step 2: (Optional) Convert to PDF
  → Use pandoc: pandoc report.md -o report.pdf
  → Or: Online converter (pandoc.org/try)

Step 3: Share with stakeholders
  → Email as attachment
  → Share in Git/Sharepoint
  → Print for meetings

Step 4: Collect feedback
  → Ask: Can you understand the results?
  → Ask: Do you want different sections?
  → Ask: Are there questions about predictions?

Step 5: Update next time
  → Keep template for next model
  → Note what worked/didn't work
  → Refine sections based on feedback
```

---

## BONUS: ONE-PAGE SUMMARY FOR BUSY PEOPLE

If someone asks "Can you summarize your model in one page?":

**Copy this template:**

```markdown
# [Model Name] - 1-Page Summary

**What:** XGBoost classifier distinguishing [CLASS_A] from [CLASS_B]
**Data:** [N_TRAIN] training samples, [N_TEST] test samples
**Features:** [N_FEATURES] protein measurements

**Results:**
- Accuracy: [VALUE]% 
- AUC: [VALUE]
- Sensitivity: [VALUE]% (catches [CLASS_A])
- Specificity: [VALUE]% (confirms [CLASS_B])

**Top 5 Proteins:**
1. [PROTEIN]: [INTERPRETATION]
2. [PROTEIN]: [INTERPRETATION]
3. [PROTEIN]: [INTERPRETATION]
4. [PROTEIN]: [INTERPRETATION]
5. [PROTEIN]: [INTERPRETATION]

**Decision Rule:**
If probability > [THRESHOLD], predict [CLASS_A]
Otherwise predict [CLASS_B]

**Use Case:** [WHY THIS MATTERS]
**Limitations:** [KEY CAVEATS]
```

**Print & paste - Done!**

---

## EMERGENCY REFERENCE

**"What number should go here?"**

```
Location on Page    → Check This File              → What It Contains
─────────────────────────────────────────────────────────────────────
Accuracy/AUC        → evaluation_results_[TS].tsv → Top metrics
Sample size         → train_matrix_[TS].tsv       → Row count
Feature count       → best_params_[TS].tsv        → Protein count
Confusion matrix    → confusion_matrix_[TS].tsv   → TP/TN/FP/FN
Cross-validation    → proteoboostr_[TS].log       → CV accuracy per fold
Date                → System clock / your calendar → Today's date
Hyperparameters     → best_params_[TS].tsv        → All 8 parameters
Probability values  → predicted_prob_[TS].tsv     → Per-sample scores
```

---

## FINAL CHECKLIST

```
Before you hit "send", ask yourself:

□ Can a non-scientist understand what problem this solves?
□ Can a clinician understand the decision rule?
□ Can a researcher reproduce this analysis?
□ Are the limitations clearly stated?
□ Is the data quality discussed?
□ Are caveats mentioned (not just strengths)?
□ Did I proofread for typos?
□ Are all images included and visible?
□ Is my conclusion honest (not overclaimed)?
□ Would I stake my reputation on these results?

If YES to all → Ready to share! 🎉
If NO to any → Fix before sharing ✏️
```

---

**Reference Sheet Version:** 1.0  
**Created:** February 10, 2026  
**For:** ProteoBoostR Model Reporting System

**Keep this card handy! Print it and tape to your desk while writing reports.**
