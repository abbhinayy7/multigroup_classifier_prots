"""Utilities for multiclass ProteoBoostR workflow
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def read_annotation(path):
    return pd.read_csv(path, sep='\t')


def read_protein(path):
    return pd.read_csv(path, sep='\t', index_col=0)


def merge_annot_prot(annot, prot, annotcol=None):
    # annot: DataFrame with sample_id in first column
    if 'sample_id' not in annot.columns:
        annot = annot.rename(columns={annot.columns[0]: 'sample_id'})
    df = prot.transpose().reset_index().rename(columns={'index': 'sample_id'})
    if annotcol:
        merged = pd.merge(annot, df, on='sample_id', how='inner')
    else:
        merged = pd.merge(annot, df, on='sample_id', how='inner')
    return merged


def preprocess_multiclass(df, label_col, control_label=None, classes=None):
    # If control_label is provided, exclude it or treat it as a class per user
    df = df.copy()
    df = df[~df[label_col].isna()].reset_index(drop=True)
    if classes:
        df = df[df[label_col].isin(classes)].reset_index(drop=True)
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df[label_col].astype(str))
    feat_cols = [c for c in df.columns if c not in ('sample_id', label_col, 'label_enc')]
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce')
    y = df['label_enc'].values
    return df, X, y, le, feat_cols


def resolve_output_dir(output):
    if output is None:
        return None
    os.makedirs(output, exist_ok=True)
    return output


def write_tsv(df, path):
    df.to_csv(path, sep='\t', index=False)


def safe_timestamp():
    import pandas as pd
    return pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
