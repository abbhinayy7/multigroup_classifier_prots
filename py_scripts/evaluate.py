#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from utils import do_merge_static, preprocess_data, resolve_output_dir, read_best_threshold


def youden_threshold(y_true, y_scores):
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    j = tpr - fpr
    best_idx = np.where(j == np.max(j))[0]
    if len(best_idx) > 1:
        # pick threshold closest to 0.5
        thr_candidates = thr[best_idx]
        return float(thr_candidates[np.argmin(np.abs(thr_candidates - 0.5))])
    return float(thr[best_idx[0]])


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved XGBoost model')
    parser.add_argument('--model', required=True)
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--protein', required=True)
    parser.add_argument('--annotcol', required=True)
    parser.add_argument('--neg', required=True)
    parser.add_argument('--pos', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    annot = pd.read_csv(args.annotation, sep='\t')
    prot = pd.read_csv(args.protein, sep='\t', index_col=0, check_names=False)

    merged = do_merge_static(annot, prot, args.annotcol)
    df = preprocess_data(merged, args.annotcol, args.neg, args.pos)
    feat_cols = [c for c in df.columns if c not in ('sample_id', args.annotcol)]

    X = df[feat_cols].values
    y = (df[args.annotcol].cat.codes).values

    model = xgb.Booster()
    model.load_model(args.model)
    dtest = xgb.DMatrix(X)
    preds = model.predict(dtest)

    # ROC and threshold
    try:
        best_thr = youden_threshold(y, preds)
    except Exception:
        best_thr = np.nan

    pred_labels = (preds > best_thr).astype(int) if not np.isnan(best_thr) else (preds > 0.5).astype(int)

    acc = accuracy_score(y, pred_labels)
    sens = recall_score(y, pred_labels)
    tn, fp, fn, tp = confusion_matrix(y, pred_labels).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)

    outdir = resolve_output_dir(args.output)
    if outdir is None:
        print('Invalid output path (Docker mode expects folder name only). Exiting.')
        return
    os.makedirs(outdir, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')

    pd.DataFrame({'sample_id': df['sample_id'], 'predicted_probability': preds}).to_csv(
        os.path.join(outdir, f'predicted_probabilities_{ts}.tsv'), sep='\t', index=False)

    results_df = pd.DataFrame([{
        'Accuracy': acc,
        'Sensitivity': sens,
        'Specificity': spec,
        'AUC': roc_auc,
        'Best_Threshold': best_thr
    }])
    results_df.to_csv(os.path.join(outdir, f'evaluation_results_{ts}.tsv'), sep='\t', index=False)

    cm = confusion_matrix(y, pred_labels)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(outdir, f'confusion_matrix_{ts}.tsv'), sep='\t', index=False)

    # ROC plot
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], linestyle='--', color='grey')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(outdir, f'roc_curve_{ts}.png'))
    plt.close()

    print('Evaluation written to', outdir)


if __name__ == '__main__':
    main()
