#!/usr/bin/env python3
"""
Binary vs Multigroup Classification Comparison
ProteoBoostR Test Suite
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("PROTEOBOOSTR: BINARY VS MULTIGROUP CLASSIFICATION TEST")
print("="*80)

# ============================================================================
# PART 1: BINARY CLASSIFICATION (GBM Dataset)
# ============================================================================

print("\n[PART 1] BINARY CLASSIFICATION TEST")
print("-"*80)
print("Dataset: GBM Werner (Subtype Classification)")

try:
    # Load data
    annot_b = pd.read_csv('GBM_testcase/Werner_annot.tsv', sep='\t')
    prot_b = pd.read_csv('GBM_testcase/Werner_data.tsv', sep='\t', index_col=0)
    
    print("Annotation shape: {}".format(annot_b.shape))
    print("Protein matrix shape: {}".format(prot_b.shape))
    
    # Prepare data
    prot_t = prot_b.T.reset_index()
    prot_t.columns = ['sample_id'] + list(prot_b.index)
    df = annot_b.merge(prot_t, on='sample_id', how='inner')
    
    # Binary classification: subtype1 vs others
    df = df[df['subtypes'].isin(['subtype1', 'subtype2+3+4'])]
    y_b = (df['subtypes'] == 'subtype2+3+4').astype(int).values
    X_b = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(df.iloc[:, 2:].mean())
    
    print("Samples: {} | Classes: 2 (subtype1 vs others)".format(len(y_b)))
    print("Features: {}".format(X_b.shape[1]))
    print("Class distribution: {} vs {}".format((y_b==0).sum(), (y_b==1).sum()))
    
    # Split
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y_b, test_size=0.25, random_state=42, stratify=y_b)
    
    print("Train: {} | Test: {}".format(len(y_train_b), len(y_test_b)))
    
    # Bayesian optimization
    print("\nOptimizing hyperparameters (binary)...")
    def binary_obj(eta, max_depth, subsample, colsample_bytree, min_child_weight, gamma, alpha, lmbda):
        params = {
            'eta': eta, 'max_depth': int(max_depth), 'subsample': subsample,
            'colsample_bytree': colsample_bytree, 'min_child_weight': int(min_child_weight),
            'gamma': gamma, 'alpha': alpha, 'lambda': lmbda,
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42
        }
        dtrain = xgb.DMatrix(X_train_b, label=y_train_b)
        cv = xgb.cv(params, dtrain, num_boost_round=300, nfold=5, 
                    early_stopping_rounds=50, verbose_eval=False)
        return cv['test-auc-mean'].iloc[-1]
    
    opt_b = BayesianOptimization(binary_obj,
        {'eta': (0.01, 0.3), 'max_depth': (1, 10), 'subsample': (0.5, 1.0),
         'colsample_bytree': (0.5, 1.0), 'min_child_weight': (1, 10),
         'gamma': (0, 5), 'alpha': (0, 10), 'lmbda': (0, 10)},
        random_state=42)
    opt_b.maximize(init_points=8, n_iter=5)
    
    best_params_b = opt_b.max['params']
    print("Best AUC: {:.4f}".format(opt_b.max['target']))
    
    # Train final model
    params_b = {
        'eta': best_params_b['eta'],
        'max_depth': int(best_params_b['max_depth']),
        'subsample': best_params_b['subsample'],
        'colsample_bytree': best_params_b['colsample_bytree'],
        'min_child_weight': int(best_params_b['min_child_weight']),
        'gamma': best_params_b['gamma'],
        'alpha': best_params_b['alpha'],
        'lambda': best_params_b['lmbda'],
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(X_train_b, label=y_train_b)
    dtest = xgb.DMatrix(X_test_b, label=y_test_b)
    model_b = xgb.train(params_b, dtrain, num_boost_round=1000,
                        evals=[(dtest, 'test')], early_stopping_rounds=100,
                        verbose_eval=False)
    
    # Predict
    y_pred_b = model_b.predict(dtest)
    y_class_b = (y_pred_b >= 0.5).astype(int)
    
    # Metrics
    acc_b = accuracy_score(y_test_b, y_class_b)
    sens_b = recall_score(y_test_b, y_class_b)
    spec_b = recall_score(y_test_b, y_class_b, pos_label=0)
    prec_b = precision_score(y_test_b, y_class_b)
    f1_b = f1_score(y_test_b, y_class_b)
    auc_b = roc_auc_score(y_test_b, y_pred_b)
    boost_b = model_b.best_iteration
    
    print("\nBINARY MODEL RESULTS:")
    print("  Accuracy:    {:.4f}".format(acc_b))
    print("  Sensitivity: {:.4f}".format(sens_b))
    print("  Specificity: {:.4f}".format(spec_b))
    print("  Precision:   {:.4f}".format(prec_b))
    print("  F1-Score:    {:.4f}".format(f1_b))
    print("  AUC-ROC:     {:.4f}".format(auc_b))
    print("  Boost Rounds: {}".format(boost_b))
    
    binary_ok = True
    
except Exception as e:
    print("ERROR in binary classification: {}".format(str(e)))
    binary_ok = False
    acc_b = prec_b = sens_b = spec_b = f1_b = auc_b = 0.0

# ============================================================================
# PART 2: MULTIGROUP CLASSIFICATION
# ============================================================================

print("\n[PART 2] MULTIGROUP CLASSIFICATION TEST")
print("-"*80)
print("Dataset: Multigroup Test Data (3-class classification)")

try:
    # Load data
    annot_m = pd.read_csv('multigroup/test_data/annotation.tsv', sep='\t')
    prot_m = pd.read_csv('multigroup/test_data/protein.tsv', sep='\t', index_col=0)
    
    print("Annotation shape: {}".format(annot_m.shape))
    print("Protein matrix shape: {}".format(prot_m.shape))
    
    # Prepare
    classes = sorted(annot_m['label'].unique())
    print("Classes: {}".format(', '.join(classes)))
    
    prot_t = prot_m.T.reset_index()
    prot_t.columns = ['sample_id'] + list(prot_m.index)
    df = annot_m.merge(prot_t, on='sample_id', how='inner')
    
    # Encode
    class_map = {c: i for i, c in enumerate(classes)}
    y_m = df['label'].map(class_map).values
    X_m = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(df.iloc[:, 2:].mean())
    
    for cls in classes:
        count = (y_m == class_map[cls]).sum()
        print("  {}: {}".format(cls, count))
    
    print("Samples: {} | Features: {}".format(len(y_m), X_m.shape[1]))
    
    # Split
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_m, y_m, test_size=0.25, random_state=42, stratify=y_m)
    
    print("Train: {} | Test: {}".format(len(y_train_m), len(y_test_m)))
    
    # Bayesian optimization
    print("\nOptimizing hyperparameters (multigroup)...")
    def multi_obj(eta, max_depth, subsample, colsample_bytree, min_child_weight, gamma, alpha, lmbda):
        params = {
            'eta': eta, 'max_depth': int(max_depth), 'subsample': subsample,
            'colsample_bytree': colsample_bytree, 'min_child_weight': int(min_child_weight),
            'gamma': gamma, 'alpha': alpha, 'lambda': lmbda,
            'objective': 'multi:softmax', 'num_class': len(classes),
            'eval_metric': 'mlogloss', 'seed': 42
        }
        dtrain = xgb.DMatrix(X_train_m, label=y_train_m)
        cv = xgb.cv(params, dtrain, num_boost_round=300, nfold=5,
                    early_stopping_rounds=50, verbose_eval=False)
        return 1.0 - (cv['test-mlogloss-mean'].iloc[-1] / np.log(len(classes)))
    
    opt_m = BayesianOptimization(multi_obj,
        {'eta': (0.01, 0.3), 'max_depth': (1, 10), 'subsample': (0.5, 1.0),
         'colsample_bytree': (0.5, 1.0), 'min_child_weight': (1, 10),
         'gamma': (0, 5), 'alpha': (0, 10), 'lmbda': (0, 10)},
        random_state=42)
    opt_m.maximize(init_points=8, n_iter=5)
    
    best_params_m = opt_m.max['params']
    print("Best optimization score: {:.4f}".format(opt_m.max['target']))
    
    # Train final model
    params_m = {
        'eta': best_params_m['eta'],
        'max_depth': int(best_params_m['max_depth']),
        'subsample': best_params_m['subsample'],
        'colsample_bytree': best_params_m['colsample_bytree'],
        'min_child_weight': int(best_params_m['min_child_weight']),
        'gamma': best_params_m['gamma'],
        'alpha': best_params_m['alpha'],
        'lambda': best_params_m['lmbda'],
        'objective': 'multi:softmax',
        'num_class': len(classes),
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(X_train_m, label=y_train_m)
    dtest = xgb.DMatrix(X_test_m, label=y_test_m)
    model_m = xgb.train(params_m, dtrain, num_boost_round=1000,
                        evals=[(dtest, 'test')], early_stopping_rounds=100,
                        verbose_eval=False)
    
    # Predict
    y_pred_m = model_m.predict(dtest).astype(int)
    
    # Metrics
    acc_m = accuracy_score(y_test_m, y_pred_m)
    prec_m = precision_score(y_test_m, y_pred_m, average='weighted', zero_division=0)
    rec_m = recall_score(y_test_m, y_pred_m, average='weighted', zero_division=0)
    f1_m = f1_score(y_test_m, y_pred_m, average='weighted', zero_division=0)
    boost_m = model_m.best_iteration
    
    print("\nMULTIGROUP MODEL RESULTS:")
    print("  Accuracy:    {:.4f}".format(acc_m))
    print("  Precision:   {:.4f}".format(prec_m))
    print("  Recall:      {:.4f}".format(rec_m))
    print("  F1-Score:    {:.4f}".format(f1_m))
    print("  Boost Rounds: {}".format(boost_m))
    
    multigroup_ok = True
    
except Exception as e:
    print("ERROR in multigroup classification: {}".format(str(e)))
    multigroup_ok = False
    acc_m = prec_m = rec_m = f1_m = 0.0

# ============================================================================
# PART 3: COMPARISON
# ============================================================================

print("\n[PART 3] SIDE-BY-SIDE COMPARISON")
print("-"*80)

if binary_ok and multigroup_ok:
    print("\nPERFORMANCE METRICS COMPARISON:")
    print("\n{:<25} {:<20} {:<20}".format("Metric", "Binary", "Multigroup"))
    print("-"*65)
    print("{:<25} {:<20.4f} {:<20.4f}".format("Accuracy", acc_b, acc_m))
    print("{:<25} {:<20.4f} {:<20.4f}".format("Precision", prec_b, prec_m))
    print("{:<25} {:<20.4f} {:<20.4f}".format("Recall/Sensitivity", sens_b, rec_m))
    print("{:<25} {:<20.4f} {:<20.4f}".format("F1-Score", f1_b, f1_m))
    print("{:<25} {:<20.4f} {:<20}".format("AUC-ROC", auc_b, "N/A (3-class)"))
    
    print("\nMODEL COMPLEXITY COMPARISON:")
    print("{:<25} {:<20} {:<20}".format("Characteristic", "Binary", "Multigroup"))
    print("-"*65)
    print("{:<25} {:<20} {:<20}".format("Classes", "2", str(len(classes))))
    print("{:<25} {:<20} {:<20}".format("Decision Boundaries", "1", str(len(classes))))
    print("{:<25} {:<20} {:<20}".format("Boosting Rounds", str(boost_b), str(boost_m)))
    
    print("\nKEY FINDINGS:")
    
    if acc_b > acc_m:
        diff = (acc_b - acc_m) * 100
        print("  1. Binary model is {:.2f}% more accurate".format(diff))
        print("     -> Simpler classification is easier for ML")
    else:
        diff = (acc_m - acc_b) * 100
        print("  1. Multigroup model is {:.2f}% more accurate".format(diff))
        print("     -> Extra classes handled well by XGBoost")
    
    print("\n  2. Binary Approach:")
    print("     - Simpler model (2-class problem)")
    print("     - Single decision threshold")
    print("     - Best for YES/NO classification")
    print("     - Higher AUC and ROC curve analysis")
    
    print("\n  3. Multigroup Approach:")
    print("     - More complex model ({} classes)".format(len(classes)))
    print("     - Multiple decision boundaries per class")
    print("     - Best for CATEGORICAL classification")
    print("     - Can distinguish {} distinct categories".format(len(classes)))
    
    print("\nRECOMMENDATIONS:")
    print("  USE BINARY when:")
    print("    - You need yes/no or positive/negative decisions")
    print("    - Maximum interpretability is needed")
    print("    - Data is naturally dichotomous")
    
    print("\n  USE MULTIGROUP when:")
    print("    - You need to distinguish multiple categories")
    print("    - {} or more distinct classes exist".format(len(classes)))
    print("    - You want one model for all classifications")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
