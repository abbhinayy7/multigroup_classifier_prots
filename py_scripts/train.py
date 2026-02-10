#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from utils import preprocess_data, do_merge_static, resolve_output_dir


def xgb_cv_auc(eta, max_depth, subsample, colsample_bytree,
               min_child_weight, gamma, alpha, lam, dtrain):
    params = dict(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='auc',
        eta=float(eta),
        max_depth=int(round(max_depth)),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        min_child_weight=int(round(min_child_weight)),
        gamma=float(gamma),
        alpha=float(alpha),
        lambda=float(lam)
    )
    cv = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5,
                early_stopping_rounds=50, verbose_eval=False, stratified=True)
    best_auc = cv['test-auc-mean'].max()
    return best_auc


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost with Bayesian optimization')
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--protein', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--annotcol', required=True)
    parser.add_argument('--neg', required=True)
    parser.add_argument('--pos', required=True)
    parser.add_argument('--testsize', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    annot = pd.read_csv(args.annotation, sep='\t')
    prot = pd.read_csv(args.protein, sep='\t', index_col=0, check_names=False)

    merged = do_merge_static(annot, prot, args.annotcol)
    df = preprocess_data(merged, args.annotcol, args.neg, args.pos)

    feat_cols = [c for c in df.columns if c not in ('sample_id', args.annotcol)]
    X = df[feat_cols].values
    y = (df[args.annotcol].cat.codes).values  # 0/1

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df['sample_id'].values, test_size=args.testsize, random_state=args.seed, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)

    # bounds
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

    def black_box(eta, max_depth, subsample, colsample_bytree,
                  min_child_weight, gamma, alpha, lam):
        return xgb_cv_auc(eta, max_depth, subsample, colsample_bytree,
                          min_child_weight, gamma, alpha, lam, dtrain)

    optimizer = BayesianOptimization(f=black_box, pbounds=pbounds, random_state=args.seed)
    optimizer.maximize(init_points=5, n_iter=20)

    best = optimizer.max['params']
    best_params = dict(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='auc',
        eta=float(best['eta']),
        max_depth=int(round(best['max_depth'])),
        subsample=float(best['subsample']),
        colsample_bytree=float(best['colsample_bytree']),
        min_child_weight=int(round(best['min_child_weight'])),
        gamma=float(best['gamma']),
        alpha=float(best['alpha']),
        lambda=float(best['lam'])
    )

    # train final model on full training set
    dtrain_full = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(best_params, dtrain_full, num_boost_round=1000)

    outdir = resolve_output_dir(args.output)
    if outdir is None:
        print('Invalid output path (Docker mode expects folder name only). Exiting.')
        return
    os.makedirs(outdir, exist_ok=True)

    ts = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    model_path = os.path.join(outdir, f'xgb_model_{ts}.json')
    model.save_model(model_path)

    params_df = pd.DataFrame({'Param': list(best_params.keys()), 'Value': list(best_params.values())})
    params_df.to_csv(os.path.join(outdir, f'best_params_{ts}.tsv'), sep='\t', index=False)

    # also save train/test matrices
    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df.insert(0, 'sample_id', idx_train)
    train_df.to_csv(os.path.join(outdir, f'train_matrix_{ts}.tsv'), sep='\t', index=False)
    test_df = pd.DataFrame(X_test, columns=feat_cols)
    test_df.insert(0, 'sample_id', idx_test)
    test_df.to_csv(os.path.join(outdir, f'test_matrix_{ts}.tsv'), sep='\t', index=False)

    print('Training complete. Model and params saved to', outdir)


if __name__ == '__main__':
    main()
