#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from utils import do_merge_static, resolve_output_dir, read_best_threshold, get_model_features
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Apply trained model to new dataset')
    parser.add_argument('--model', required=True)
    parser.add_argument('--protein', required=True)
    parser.add_argument('--annotation', required=False)
    parser.add_argument('--annotcol', required=False)
    parser.add_argument('--neg', required=False)
    parser.add_argument('--pos', required=False)
    parser.add_argument('--evaltsv', required=False, help='Evaluation TSV containing Best_Threshold')
    parser.add_argument('--band', type=float, default=0.1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    annot = pd.read_csv(args.annotation, sep='\t') if args.annotation else None
    prot = pd.read_csv(args.protein, sep='\t', index_col=0, check_names=False)
    merged = do_merge_static(annot, prot, args.annotcol if args.annotcol else None)

    has_labels = (annot is not None) and (args.annotcol in merged.columns) and (annot.shape[0] > 0)

    df = merged.copy()
    if has_labels:
        # keep only rows with neg/pos
        df = df[df[args.annotcol].isin([args.neg, args.pos])]
        df[args.annotcol] = pd.Categorical(df[args.annotcol], categories=[args.neg, args.pos])

    model = xgb.Booster()
    model.load_model(args.model)
    model_feats = get_model_features(model)
    feat_cols_df = [c for c in df.columns if c not in ('sample_id', args.annotcol)]
    present = [f for f in model_feats if f in feat_cols_df]
    missing = [f for f in model_feats if f not in feat_cols_df]
    X = df[present].copy()
    for m in missing:
        X[m] = np.nan
    X = X[model_feats]
    X = X.apply(pd.to_numeric, errors='coerce')

    dtest = xgb.DMatrix(X.values)
    preds = model.predict(dtest)

    base_thr = None
    if args.evaltsv:
        base_thr = read_best_threshold(args.evaltsv)
    if pd.isna(base_thr) and hasattr(model, 'best_threshold'):
        base_thr = getattr(model, 'best_threshold')
    if base_thr is None or pd.isna(base_thr):
        base_thr = np.nan

    tmin = np.nan if np.isnan(base_thr) else max(0, base_thr - args.band)
    tmax = np.nan if np.isnan(base_thr) else min(1, base_thr + args.band)

    pred_tbl = pd.DataFrame({'sample_id': df['sample_id'], 'preds': preds})
    if has_labels:
        pred_tbl['ann'] = df[args.annotcol].values

    pred_tbl = pred_tbl.sort_values('preds', ascending=False).reset_index(drop=True)
    pred_tbl['rank'] = np.arange(1, len(pred_tbl) + 1)

    if not np.isnan(base_thr):
        pos_label = args.pos if has_labels else 'pos'
        neg_label = args.neg if has_labels else 'neg'
        pred_tbl['zone'] = pred_tbl['preds'].apply(lambda v: 'below' if v < tmin else ('above' if v > tmax else 'not classified'))
        pred_tbl['pred_label'] = pred_tbl['zone'].apply(lambda z: neg_label if z == 'below' else (pos_label if z == 'above' else None))

    outdir = resolve_output_dir(args.output)
    if outdir is None:
        print('Invalid output path (Docker mode expects folder name only). Exiting.')
        return
    os.makedirs(outdir, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')

    pred_tbl[['sample_id', 'preds'] + (['ann'] if 'ann' in pred_tbl.columns else [])].to_csv(
        os.path.join(outdir, f'predicted_probabilities_{ts}_adhoc.tsv'), sep='\t', index=False)

    # if labels present and base_thr present, compute confusion and ROC for classified samples
    if has_labels and not np.isnan(base_thr):
        kept = pred_tbl[pred_tbl['zone'] != 'not classified']
        if len(kept) > 0:
            y_true = (kept['ann'].astype('category').cat.codes).values
            y_scores = kept['preds'].values
            pred_class = (y_scores > base_thr).astype(int)
            try:
                cm = confusion_matrix(y_true, pred_class)
                cm_df = pd.DataFrame(cm)
                cm_df.to_csv(os.path.join(outdir, f'confusion_matrix_{ts}_adhoc.tsv'), sep='\t', index=False)
            except Exception:
                pass
            if len(np.unique(y_true)) == 2 and len(kept) >= 2:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6,6))
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                plt.plot([0,1],[0,1], linestyle='--', color='grey')
                plt.savefig(os.path.join(outdir, f'roc_curve_{ts}_adhoc.png'))
                plt.close()

    print('Ad-hoc application outputs written to', outdir)


if __name__ == '__main__':
    main()
