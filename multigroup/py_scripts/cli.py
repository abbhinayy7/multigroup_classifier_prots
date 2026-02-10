#!/usr/bin/env python3
"""Multigroup CLI: train/evaluate/apply for multiclass proteomics classification"""
import argparse
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import joblib
from utils import read_annotation, read_protein, merge_annot_prot, preprocess_multiclass, resolve_output_dir, write_tsv, safe_timestamp

# logger
logger = logging.getLogger('multigroup')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def plot_multiclass_roc(y_true, y_score, le, outpath, title='ROC (One-vs-Rest)'):
    # y_true: shape (n_samples,) integer labels
    # y_score: shape (n_samples, n_classes) probability per class
    n_classes = y_score.shape[1]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10')
    all_fpr = np.unique(np.concatenate([roc_curve((y_true == i).astype(int), y_score[:, i])[0] for i in range(n_classes)]))
    # Compute interpolation for macro-average
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
        plt.plot(fpr, tpr, color=colors(i), lw=2, label=f'{le.inverse_transform([i])[0]} (AUC={auc(fpr,tpr):.3f})')
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    roc_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', lw=3, label=f'Macro Avg (AUC={roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='grey', linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_pairwise_roc(y_true, y_score, le, outdir, ts_prefix):
    """Compute and save 1-vs-1 ROC plots for every pair of classes.
    Returns a DataFrame with pairwise AUCs.
    """
    n_classes = y_score.shape[1]
    pairs = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            # subset to samples belonging to class i or j
            mask = np.isin(y_true, [i, j])
            if mask.sum() < 10:
                # skip tiny pairs
                continue
            y_bin = (y_true[mask] == i).astype(int)
            p_i = y_score[mask, i]
            p_j = y_score[mask, j]
            score = p_i / (p_i + p_j + 1e-12)
            fpr, tpr, _ = roc_curve(y_bin, score)
            pair_auc = auc(fpr, tpr)
            # plot
            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, lw=2)
            plt.plot([0,1], [0,1], color='grey', linestyle=':')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            a_name = le.inverse_transform([i])[0]
            b_name = le.inverse_transform([j])[0]
            plt.title(f'ROC: {a_name} vs {b_name} (AUC={pair_auc:.3f})')
            plt.grid(alpha=0.3)
            fname = os.path.join(outdir, f'roc_pair_{a_name}_vs_{b_name}_{ts_prefix}.png')
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            pairs.append({'class_A': a_name, 'class_B': b_name, 'auc': pair_auc})
    return pd.DataFrame(pairs)


def plot_ranked_multiclass(pred_tbl, outpath, title='Ranked samples (max prob)'):
    # pred_tbl: DataFrame with sample_id, true_label, max_prob, predicted_label
    plt.figure(figsize=(13,8))
    colors = plt.cm.get_cmap('tab10')
    unique_labels = pred_tbl['true_label'].unique()
    for i, lab in enumerate(sorted(unique_labels)):
        df = pred_tbl[pred_tbl['true_label'] == lab]
        plt.scatter(df['rank'], df['max_prob'], label=f'{lab} (n={len(df)})', color=colors(i), edgecolors='black')
    plt.xlabel('Rank (desc)')
    plt.ylabel('Max predicted probability')
    plt.title(title)
    plt.ylim(-0.01, 1.01)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def train_command(args):
    annot = read_annotation(args.annotation)
    prot = read_protein(args.protein)
    logger.info('Merging annotation + protein...')
    merged = merge_annot_prot(annot, prot, args.annotcol)
    df, X, y, le, feat_cols = preprocess_multiclass(merged, args.annotcol, classes=args.classes)
    n_classes = len(np.unique(y))
    logger.info('Detected %d classes: %s', n_classes, list(le.classes_))
    # basic checks
    if n_classes < 2:
        raise ValueError('Need at least two classes for training')
    counts = np.bincount(y)
    if counts[counts>0].min() < 3:
        logger.warning('One or more classes have very few samples (min=%d). Consider collecting more samples or using resampling.', counts[counts>0].min())

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, df['sample_id'].values, test_size=args.testsize, stratify=y, random_state=args.seed)

    # optimize hyperparams (maximize cross-val f1_macro)
    dtrain = X_train
    ytr = y_train

    def xgb_cv(f_learning_rate, f_max_depth, f_subsample, f_colsample, f_min_child_weight, f_gamma, f_reg_alpha, f_reg_lambda):
        params = dict(
            learning_rate=float(f_learning_rate),
            max_depth=int(round(f_max_depth)),
            subsample=float(f_subsample),
            colsample_bytree=float(f_colsample),
            min_child_weight=int(round(f_min_child_weight)),
            gamma=float(f_gamma),
            reg_alpha=float(f_reg_alpha),
            reg_lambda=float(f_reg_lambda),
            n_estimators=500,
            objective='multi:softprob'
        )
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
        # adapt number of splits to smallest class size
        try:
            counts = np.bincount(ytr)
            min_class_count = counts[counts > 0].min()
            n_splits_cv = min(5, int(min_class_count))
            n_splits_cv = max(2, n_splits_cv)
            cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=args.seed)
            scores = cross_val_score(model, dtrain, ytr, cv=cv, scoring='f1_macro')
            return float(scores.mean())
        except Exception:
            return 0.0

    pbounds = {
        'f_learning_rate': (0.001, 0.5),
        'f_max_depth': (2, 12),
        'f_subsample': (0.4, 1.0),
        'f_colsample': (0.3, 1.0),
        'f_min_child_weight': (1, 10),
        'f_gamma': (0.0, 5.0),
        'f_reg_alpha': (0.0, 10.0),
        'f_reg_lambda': (0.0, 10.0),
    }

    optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=args.seed)
    optimizer.maximize(init_points=args.init_points, n_iter=args.n_iter)
    best = optimizer.max['params']
    best_params = {
        'learning_rate': float(best['f_learning_rate']),
        'max_depth': int(round(best['f_max_depth'])),
        'subsample': float(best['f_subsample']),
        'colsample_bytree': float(best['f_colsample']),
        'min_child_weight': int(round(best['f_min_child_weight'])),
        'gamma': float(best['f_gamma']),
        'reg_alpha': float(best['f_reg_alpha']),
        'reg_lambda': float(best['f_reg_lambda']),
        'n_estimators': 1000,
        'objective': 'multi:softprob'
    }
    logger.info('Best params: %s', best_params)

    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    model.fit(X_train, y_train)

    outdir = resolve_output_dir(args.output)
    ts = safe_timestamp()
    model_path = os.path.join(outdir, f'xgb_model_{ts}.joblib')
    joblib.dump({'model': model, 'label_encoder': le, 'features': feat_cols}, model_path)
    logger.info('Model saved: %s', model_path)

    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df.insert(0, 'sample_id', idx_train)
    write_tsv(train_df, os.path.join(outdir, f'train_matrix_{ts}.tsv'))
    test_df = pd.DataFrame(X_test, columns=feat_cols)
    test_df.insert(0, 'sample_id', idx_test)
    write_tsv(test_df, os.path.join(outdir, f'test_matrix_{ts}.tsv'))

    # best params
    bp = pd.DataFrame({'param': list(best_params.keys()), 'value': list(best_params.values())})
    write_tsv(bp, os.path.join(outdir, f'best_params_{ts}.tsv'))
    logger.info('Training outputs written to %s', outdir)


def evaluate_command(args):
    data = read_annotation(args.annotation)
    prot = read_protein(args.protein)
    merged = merge_annot_prot(data, prot, args.annotcol)
    df, X, y, le, feat_cols = preprocess_multiclass(merged, args.annotcol, classes=args.classes)

    model_data = joblib.load(args.model)
    model = model_data['model']
    le_model = model_data['label_encoder']
    preds = model.predict(X)
    probs = model.predict_proba(X)

    # classification report + confusion matrix
    report = classification_report(y, preds, target_names=list(le.classes_), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    outdir = resolve_output_dir(args.output)
    ts = safe_timestamp()
    write_tsv(report_df.reset_index(), os.path.join(outdir, f'evaluation_report_{ts}.tsv'))

    cm = confusion_matrix(y, preds)
    write_tsv(pd.DataFrame(cm), os.path.join(outdir, f'confusion_matrix_{ts}.tsv'))

    # ROC
    if getattr(args, 'roc_mode', 'ovr') == 'ovr':
        plot_multiclass_roc(y, probs, le, os.path.join(outdir, f'roc_curve_{ts}.png'))
    else:
        pairwise_df = plot_pairwise_roc(y, probs, le, outdir, ts)
        if not pairwise_df.empty:
            write_tsv(pairwise_df, os.path.join(outdir, f'pairwise_aucs_{ts}.tsv'))
        else:
            logger.info('No pairwise ROC plots were generated (pairs too small)')

    # ranked (max prob)
    max_prob = probs.max(axis=1)
    pred_label_names = [le.inverse_transform([p])[0] for p in preds]
    true_label_names = [le.inverse_transform([t])[0] for t in y]
    pred_tbl = pd.DataFrame({'sample_id': df['sample_id'], 'true_label': true_label_names, 'pred_label': pred_label_names, 'max_prob': max_prob})
    pred_tbl = pred_tbl.sort_values('max_prob', ascending=False).reset_index(drop=True)
    pred_tbl['rank'] = np.arange(1, len(pred_tbl) + 1)
    write_tsv(pred_tbl, os.path.join(outdir, f'predicted_probabilities_{ts}.tsv'))
    plot_ranked_multiclass(pred_tbl, os.path.join(outdir, f'predicted_samples_{ts}.png'))

    logger.info('Evaluation outputs written to %s', outdir)


def apply_command(args):
    prot = read_protein(args.protein)
    if args.annotation:
        annot = read_annotation(args.annotation)
    else:
        annot = pd.DataFrame({'sample_id': prot.columns})
    merged = merge_annot_prot(annot, prot, args.annotcol)
    df, X, y, le, feat_cols = preprocess_multiclass(merged, args.annotcol, classes=args.classes)

    model_data = joblib.load(args.model)
    model = model_data['model']
    model_feats = model_data['features']
    # align
    present = [f for f in model_feats if f in X.columns]
    missing = [f for f in model_feats if f not in X.columns]
    for m in missing:
        X[m] = np.nan
    X = X[model_feats]
    probs = model.predict_proba(X)
    max_prob = probs.max(axis=1)
    preds = model.predict(X)
    pred_names = [model_data['label_encoder'].inverse_transform([p])[0] for p in preds]
    outdir = resolve_output_dir(args.output)
    ts = safe_timestamp()
    pred_tbl = pd.DataFrame({'sample_id': df['sample_id'], 'predicted_label': pred_names, 'max_prob': max_prob})
    pred_tbl = pred_tbl.sort_values('max_prob', ascending=False).reset_index(drop=True)
    pred_tbl['rank'] = np.arange(1, len(pred_tbl) + 1)
    write_tsv(pred_tbl, os.path.join(outdir, f'predicted_probabilities_{ts}_adhoc.tsv'))
    plot_ranked_multiclass(pred_tbl.assign(true_label=pred_tbl['predicted_label']), os.path.join(outdir, f'predicted_samples_{ts}_adhoc.png'))
    logger.info('Ad-hoc outputs written to %s', outdir)


def main():
    p = argparse.ArgumentParser(description='Multigroup ProteoBoostR CLI')
    sub = p.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--annotation', required=True)
    t.add_argument('--protein', required=True)
    t.add_argument('--annotcol', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--n_iter', type=int, default=25)
    t.add_argument('--init_points', type=int, default=5)
    t.add_argument('--testsize', type=float, default=0.3)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--classes', nargs='*', help='optional list of classes to keep')

    e = sub.add_parser('evaluate')
    e.add_argument('--model', required=True)
    e.add_argument('--annotation', required=True)
    e.add_argument('--protein', required=True)
    e.add_argument('--annotcol', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--classes', nargs='*')
    e.add_argument('--roc_mode', choices=['ovr','ovo'], default='ovr', help='ROC mode: one-vs-rest (ovr) or one-vs-one (ovo)')

    a = sub.add_parser('apply')
    a.add_argument('--model', required=True)
    a.add_argument('--protein', required=True)
    a.add_argument('--annotation', required=False)
    a.add_argument('--annotcol', required=False)
    a.add_argument('--output', required=True)
    a.add_argument('--classes', nargs='*')

    args = p.parse_args()
    if args.cmd == 'train':
        train_command(args)
    elif args.cmd == 'evaluate':
        evaluate_command(args)
    elif args.cmd == 'apply':
        apply_command(args)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
