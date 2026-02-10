#!/usr/bin/env python3
"""Unified CLI for ProteoBoostR-like workflow: train, evaluate, apply

Run as:
  python py_scripts/cli.py train --help
  python py_scripts/cli.py evaluate --help
  python py_scripts/cli.py apply --help
"""
import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from utils import do_merge_static, preprocess_data, resolve_output_dir, read_best_threshold, get_model_features


# configure root logger
logger = logging.getLogger("proteoboostr")
_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S.%f")
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
if not logger.handlers:
    logger.addHandler(_sh)
logger.setLevel(logging.INFO)


def youden_threshold(y_true, y_scores):
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    j = tpr - fpr
    best_idx = np.where(j == np.max(j))[0]
    if len(best_idx) > 1:
        thr_candidates = thr[best_idx]
        return float(thr_candidates[np.argmin(np.abs(thr_candidates - 0.5))])
    return float(thr[best_idx[0]])


def plot_ranked_samples(pred_tbl, output_path, threshold=None, title="Ranked Predicted Samples"):
    """Plot ranked prediction scores with optional color coding for labels."""
    plt.figure(figsize=(8, 6))
    
    if 'ann' in pred_tbl.columns:
        colors = {'0': 'blue', '1': 'red'}
        for label in pred_tbl['ann'].unique():
            mask = pred_tbl['ann'] == label
            plt.scatter(pred_tbl[mask]['rank'], pred_tbl[mask]['preds'], label=str(label), alpha=0.6)
    else:
        plt.scatter(pred_tbl['rank'], pred_tbl['preds'], color='blue', alpha=0.6)
    
    if threshold is not None and not np.isnan(threshold):
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    
    plt.xlabel('Sample Rank (desc prob)')
    plt.ylabel('Predicted Probability')
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def train_command(args):
    try:
        if not os.path.exists(args.annotation):
            logger.error('Annotation file not found: %s', args.annotation)
            raise SystemExit(2)
        if not os.path.exists(args.protein):
            logger.error('Protein file not found: %s', args.protein)
            raise SystemExit(2)

        logger.info('Loading annotation file: %s', args.annotation)
        annot = pd.read_csv(args.annotation, sep='\t')
        logger.info('Annotation file loaded. Rows: %d Cols: %d', annot.shape[0], annot.shape[1])

        logger.info('Loading protein matrix: %s', args.protein)
        prot = pd.read_csv(args.protein, sep='\t', index_col=0)
        logger.info('Protein matrix loaded. Dimensions: %d x %d', prot.shape[0], prot.shape[1])

        logger.info('Merging annotation and protein data...')
        merged = do_merge_static(annot, prot, args.annotcol)
        logger.info('Merging complete.')

        logger.info('Preprocessing data (filtering to %s vs %s classes)...', args.neg, args.pos)
        df = preprocess_data(merged, args.annotcol, args.neg, args.pos)
        logger.info('Preprocessed data: %d samples, %d features', df.shape[0], df.shape[1])

        feat_cols = [c for c in df.columns if c not in ('sample_id', args.annotcol)]
        X = df[feat_cols].values
        y = (df[args.annotcol].cat.codes).values

        logger.info('Performing train/test split (test_size=%.2f, seed=%d)', args.testsize, args.seed)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df['sample_id'].values, test_size=args.testsize, random_state=args.seed, stratify=y
        )
        logger.info('Train samples: %d, Test samples: %d', len(X_train), len(X_test))

        dtrain = xgb.DMatrix(X_train, label=y_train)

        def xgb_cv_auc(eta, max_depth, subsample, colsample_bytree,
                       min_child_weight, gamma, alpha, lam):
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'eta': float(eta),
                'max_depth': int(round(max_depth)),
                'subsample': float(subsample),
                'colsample_bytree': float(colsample_bytree),
                'min_child_weight': int(round(min_child_weight)),
                'gamma': float(gamma),
                'alpha': float(alpha),
                'lambda': float(lam)
            }
            cv = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5,
                        early_stopping_rounds=50, verbose_eval=False, stratified=True)
            return cv['test-auc-mean'].max()

        pbounds = {
            'eta': (0.01, 0.3),
            'max_depth': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0.0, 5.0),
            'alpha': (0.0, 10.0),
            'lam': (0.0, 10.0)
        }

        logger.info('Starting Bayesian Optimization (init_points=%d, n_iter=%d)...', args.init_points, args.n_iter)
        try:
            optimizer = BayesianOptimization(f=xgb_cv_auc, pbounds=pbounds, random_state=args.seed)
            optimizer.maximize(init_points=args.init_points, n_iter=args.n_iter)
        except Exception as e:
            logger.exception('Bayesian optimization failed: %s', e)
            raise SystemExit(3)

        best = optimizer.max['params']
        best_auc = optimizer.max['target']
        logger.info('Bayesian Optimization done. Best AUC = %.4f', best_auc)

        best_params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': float(best['eta']),
            'max_depth': int(round(best['max_depth'])),
            'subsample': float(best['subsample']),
            'colsample_bytree': float(best['colsample_bytree']),
            'min_child_weight': int(round(best['min_child_weight'])),
            'gamma': float(best['gamma']),
            'alpha': float(best['alpha']),
            'lambda': float(best['lam'])
        }

        logger.info('Final training columns used for XGB: %s', ', '.join(feat_cols))
        dtrain_full = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(best_params, dtrain_full, num_boost_round=1000)

        outdir = resolve_output_dir(args.output)
        if outdir is None:
            logger.error('Invalid output path (Docker mode expects folder name only).')
            raise SystemExit(2)
        os.makedirs(outdir, exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')

        # add file logger for this run
        fh = logging.FileHandler(os.path.join(outdir, f'proteoboostr_{ts}.log'))
        fh.setFormatter(_fmt)
        logger.addHandler(fh)

        model_path = os.path.join(outdir, f'xgb_model_{ts}.json')
        try:
            model.save_model(model_path)
            logger.info('Model saved to: %s', model_path)

            params_df = pd.DataFrame({'Param': list(best_params.keys()), 'Value': list(best_params.values())})
            params_df.to_csv(os.path.join(outdir, f'best_params_{ts}.tsv'), sep='\t', index=False)
            logger.info('Best hyperparameters written.')

            train_df = pd.DataFrame(X_train, columns=feat_cols)
            train_df.insert(0, 'sample_id', idx_train)
            train_df.to_csv(os.path.join(outdir, f'train_matrix_{ts}.tsv'), sep='\t', index=False)

            test_df = pd.DataFrame(X_test, columns=feat_cols)
            test_df.insert(0, 'sample_id', idx_test)
            test_df.to_csv(os.path.join(outdir, f'test_matrix_{ts}.tsv'), sep='\t', index=False)

            logger.info('Final XGBoost model trained.')
        except Exception as e:
            logger.exception('Failed to save model or outputs: %s', e)
            raise SystemExit(4)
        finally:
            # remove file handler from logger to avoid duplicate logs in later runs
            logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    except SystemExit:
        raise
    except Exception as e:
        logger.exception('Training failed: %s', e)
        raise SystemExit(1)


def evaluate_command(args):
    try:
        if not os.path.exists(args.model):
            logger.error('Model file not found: %s', args.model)
            raise SystemExit(2)
        if not os.path.exists(args.annotation):
            logger.error('Annotation file not found: %s', args.annotation)
            raise SystemExit(2)
        if not os.path.exists(args.protein):
            logger.error('Protein file not found: %s', args.protein)
            raise SystemExit(2)

        logger.info('Loading annotation file: %s', args.annotation)
        annot = pd.read_csv(args.annotation, sep='\t')
        logger.info('Annotation file loaded. Rows: %d Cols: %d', annot.shape[0], annot.shape[1])

        logger.info('Loading protein matrix: %s', args.protein)
        prot = pd.read_csv(args.protein, sep='\t', index_col=0)
        logger.info('Protein matrix loaded. Dimensions: %d x %d', prot.shape[0], prot.shape[1])

        logger.info('Merging annotation and protein data...')
        merged = do_merge_static(annot, prot, args.annotcol)
        logger.info('Merging complete.')

        logger.info('Preprocessing data (filtering to %s vs %s classes)...', args.neg, args.pos)
        df = preprocess_data(merged, args.annotcol, args.neg, args.pos)
        feat_cols = [c for c in df.columns if c not in ('sample_id', args.annotcol)]

        X = df[feat_cols].values
        y = (df[args.annotcol].cat.codes).values

        logger.info('Evaluating model on test data...')
        model = xgb.Booster()
        try:
            model.load_model(args.model)
            logger.info('Model loaded successfully.')
        except Exception as e:
            logger.exception('Failed to load model: %s', e)
            raise SystemExit(3)

        logger.info('Test model columns used for XGB: %s', ', '.join(feat_cols))
        dtest = xgb.DMatrix(X)
        preds = model.predict(dtest)

        try:
            best_thr = youden_threshold(y, preds)
        except Exception:
            best_thr = np.nan

        pred_labels = (preds > best_thr).astype(int) if not np.isnan(best_thr) else (preds > 0.5).astype(int)

        acc = accuracy_score(y, pred_labels)
        sens = recall_score(y, pred_labels)
        prec = precision_score(y, pred_labels)
        f1 = f1_score(y, pred_labels)
        tn, fp, fn, tp = confusion_matrix(y, pred_labels).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fpr, tpr, _ = roc_curve(y, preds)
        roc_auc = auc(fpr, tpr)

        outdir = resolve_output_dir(args.output)
        if outdir is None:
            logger.error('Invalid output path (Docker mode expects folder name only).')
            raise SystemExit(2)
        os.makedirs(outdir, exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')

        # attach file logger
        fh = logging.FileHandler(os.path.join(outdir, f'proteoboostr_{ts}.log'))
        fh.setFormatter(_fmt)
        logger.addHandler(fh)

        try:
            logger.info('Writing outputs to %s', outdir)

            # predicted probabilities
            pd.DataFrame({'sample_id': df['sample_id'], 'predicted_probability': preds}).to_csv(
                os.path.join(outdir, f'predicted_probabilities_{ts}.tsv'), sep='\t', index=False)
            logger.info('Predicted probabilities written.')

            # evaluation results
            results_df = pd.DataFrame([{
                'Accuracy': acc,
                'Sensitivity': sens,
                'Precision': prec,
                'F1': f1,
                'Specificity': spec,
                'AUC': roc_auc,
                'Best_Threshold': best_thr
            }])
            results_df.to_csv(os.path.join(outdir, f'evaluation_results_{ts}.tsv'), sep='\t', index=False)
            logger.info('Evaluation results written.')

            # confusion matrix
            cm = confusion_matrix(y, pred_labels)
            cm_df = pd.DataFrame(cm)
            cm_df.to_csv(os.path.join(outdir, f'confusion_matrix_{ts}.tsv'), sep='\t', index=False)
            logger.info('Confusion matrix written.')

            # ranked samples plot
            pred_tbl = pd.DataFrame({
                'sample_id': df['sample_id'],
                'ann': df[args.annotcol].values,
                'preds': preds
            }).sort_values('preds', ascending=False).reset_index(drop=True)
            pred_tbl['rank'] = np.arange(1, len(pred_tbl) + 1)
            
            plot_ranked_samples(pred_tbl, os.path.join(outdir, f'predicted_samples_{ts}.png'), 
                              threshold=best_thr, title='Ranked Predicted Samples (Test Set)')
            logger.info('Ranked samples plot written.')

            # ROC curve
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('1 - Specificity')
            plt.ylabel('Sensitivity')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'roc_curve_{ts}.png'), dpi=100)
            plt.close()
            logger.info('ROC curve written as PNG.')

            logger.info('Test evaluation complete.')
            logger.info('Test outputs written.')
        except Exception as e:
            logger.exception('Failed while writing evaluation outputs: %s', e)
            raise SystemExit(4)
        finally:
            logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    except SystemExit:
        raise
    except Exception as e:
        logger.exception('Evaluation failed: %s', e)
        raise SystemExit(1)


def apply_command(args):
    try:
        if args.annotation and not os.path.exists(args.annotation):
            logger.error('Annotation file not found: %s', args.annotation)
            raise SystemExit(2)
        if not os.path.exists(args.protein):
            logger.error('Protein file not found: %s', args.protein)
            raise SystemExit(2)

        if args.annotation:
            logger.info('Loading annotation file: %s', args.annotation)
            annot = pd.read_csv(args.annotation, sep='\t')
            logger.info('Annotation file loaded. Rows: %d Cols: %d', annot.shape[0], annot.shape[1])
        else:
            annot = None
            logger.info('No annotation file provided.')

        logger.info('Loading protein matrix: %s', args.protein)
        prot = pd.read_csv(args.protein, sep='\t', index_col=0)
        logger.info('Protein matrix loaded. Dimensions: %d x %d', prot.shape[0], prot.shape[1])

        logger.info('Merging annotation and protein data...')
        merged = do_merge_static(annot, prot, args.annotcol if args.annotcol else None)
        logger.info('Merging complete.')

        has_labels = (annot is not None) and (args.annotcol in merged.columns) and (annot.shape[0] > 0)

        df = merged.copy()
        if has_labels:
            df = df[df[args.annotcol].isin([args.neg, args.pos])]
            df[args.annotcol] = pd.Categorical(df[args.annotcol], categories=[args.neg, args.pos])

        model = xgb.Booster()
        try:
            if not os.path.exists(args.model):
                logger.error('Model file not found: %s', args.model)
                raise SystemExit(2)
            model.load_model(args.model)
            logger.info('Model loaded successfully.')
        except Exception as e:
            logger.exception('Failed to load model: %s', e)
            raise SystemExit(3)

        model_feats = get_model_features(model)
        feat_cols_df = [c for c in df.columns if c not in ('sample_id', args.annotcol)]
        present = [f for f in model_feats if f in feat_cols_df]
        missing = [f for f in model_feats if f not in feat_cols_df]
        logger.info('Feature alignment: %d present, %d missing', len(present), len(missing))

        X = df[present].copy()
        for m in missing:
            X[m] = np.nan
        X = X[model_feats]
        X = X.apply(pd.to_numeric, errors='coerce')

        logger.info('Generating predictions...')
        dtest = xgb.DMatrix(X.values)
        preds = model.predict(dtest)

        base_thr = np.nan
        if args.evaltsv:
            logger.info('Reading base threshold from %s', args.evaltsv)
            base_thr = read_best_threshold(args.evaltsv)
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
            logger.info('Threshold banding: base=%.4f, band=%.4f, zone=[%.4f, %.4f]', base_thr, args.band, tmin, tmax)

        outdir = resolve_output_dir(args.output)
        if outdir is None:
            logger.error('Invalid output path (Docker mode expects folder name only).')
            raise SystemExit(2)
        os.makedirs(outdir, exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')

        fh = logging.FileHandler(os.path.join(outdir, f'proteoboostr_{ts}.log'))
        fh.setFormatter(_fmt)
        logger.addHandler(fh)

        try:
            logger.info('Writing ad-hoc outputs to %s', outdir)

            cols = ['sample_id', 'preds'] + (['ann'] if 'ann' in pred_tbl.columns else [])
            pred_tbl[cols].to_csv(os.path.join(outdir, f'predicted_probabilities_{ts}_adhoc.tsv'), sep='\t', index=False)
            logger.info('Ad-hoc predicted probabilities written.')

            # ranked samples plot for ad-hoc
            plot_ranked_samples(pred_tbl, os.path.join(outdir, f'predicted_samples_{ts}_adhoc.png'),
                              threshold=base_thr if not np.isnan(base_thr) else None,
                              title='Ranked Predicted Samples (Ad-hoc Application)')
            logger.info('Ranked samples plot written.')

            if has_labels and not np.isnan(base_thr):
                kept = pred_tbl[pred_tbl['zone'] != 'not classified']
                n_kept = len(kept)
                n_total = len(pred_tbl)
                logger.info('Classified samples: %d / %d (band-based threshold)', n_kept, n_total)

                if n_kept > 0:
                    y_true = (kept['ann'].astype('category').cat.codes).values
                    y_scores = kept['preds'].values
                    pred_class = (y_scores > base_thr).astype(int)

                    # confusion matrix
                    cm = confusion_matrix(y_true, pred_class)
                    pd.DataFrame(cm).to_csv(os.path.join(outdir, f'confusion_matrix_{ts}_adhoc.tsv'), sep='\t', index=False)
                    logger.info('Confusion matrix written.')

                    # evaluation metrics
                    acc = accuracy_score(y_true, pred_class)
                    sens = recall_score(y_true, pred_class)
                    prec = precision_score(y_true, pred_class)
                    f1 = f1_score(y_true, pred_class)
                    tn, fp, fn, tp = cm.ravel()
                    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                    results_df = pd.DataFrame([{
                        'Accuracy': acc,
                        'Sensitivity': sens,
                        'Precision': prec,
                        'F1': f1,
                        'Specificity': spec
                    }])
                    results_df.to_csv(os.path.join(outdir, f'evaluation_results_{ts}_adhoc.tsv'), sep='\t', index=False)
                    logger.info('Evaluation results written.')

                    # ROC curve
                    if len(np.unique(y_true)) == 2 and n_kept >= 2:
                        fpr, tpr, _ = roc_curve(y_true, y_scores)
                        roc_auc = auc(fpr, tpr)
                        plt.figure(figsize=(6, 6))
                        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                        plt.xlabel('1 - Specificity')
                        plt.ylabel('Sensitivity')
                        plt.title('ROC Curve (Ad-hoc, Classified Samples)')
                        plt.legend(loc='lower right')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(outdir, f'roc_curve_{ts}_adhoc.png'), dpi=100)
                        plt.close()
                        logger.info('ROC curve written as PNG.')

            logger.info('Ad-hoc application outputs written to: %s', outdir)
        except Exception as e:
            logger.exception('Failed while writing ad-hoc outputs: %s', e)
            raise SystemExit(4)
        finally:
            logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    except SystemExit:
        raise
    except Exception as e:
        logger.exception('Ad-hoc application failed: %s', e)
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description='ProteoBoostR Python CLI')
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('--annotation', required=True)
    p_train.add_argument('--protein', required=True)
    p_train.add_argument('--annotcol', required=True)
    p_train.add_argument('--neg', required=True)
    p_train.add_argument('--pos', required=True)
    p_train.add_argument('--output', required=True)
    p_train.add_argument('--testsize', type=float, default=0.3)
    p_train.add_argument('--seed', type=int, default=1234)
    p_train.add_argument('--n_iter', type=int, default=20, help='BayesOpt iterations')
    p_train.add_argument('--init_points', type=int, default=5, help='BayesOpt init points')

    p_eval = sub.add_parser('evaluate')
    p_eval.add_argument('--model', required=True)
    p_eval.add_argument('--annotation', required=True)
    p_eval.add_argument('--protein', required=True)
    p_eval.add_argument('--annotcol', required=True)
    p_eval.add_argument('--neg', required=True)
    p_eval.add_argument('--pos', required=True)
    p_eval.add_argument('--output', required=True)

    p_apply = sub.add_parser('apply')
    p_apply.add_argument('--model', required=True)
    p_apply.add_argument('--protein', required=True)
    p_apply.add_argument('--annotation', required=False)
    p_apply.add_argument('--annotcol', required=False)
    p_apply.add_argument('--neg', required=False)
    p_apply.add_argument('--pos', required=False)
    p_apply.add_argument('--evaltsv', required=False)
    p_apply.add_argument('--band', type=float, default=0.1)
    p_apply.add_argument('--output', required=True)

    args = parser.parse_args()
    if args.cmd == 'train':
        train_command(args)
    elif args.cmd == 'evaluate':
        evaluate_command(args)
    elif args.cmd == 'apply':
        apply_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
