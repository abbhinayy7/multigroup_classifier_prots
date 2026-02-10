import os
import pandas as pd
import numpy as np


def resolve_output_dir(user_path):
    """Resolve output dir. If a special file `ProteoBoostR.filesystem` exists, interpret as Docker mode
    and only allow folder names (no slashes). Otherwise return user_path as given."""
    config_file = "ProteoBoostR.filesystem"
    if os.path.exists(config_file):
        # Docker mode: user_path should be a simple folder name (no slashes)
        if any(sep in user_path for sep in ['/', '\\']):
            return None
        system_root = open(config_file, 'r').read().strip()
        return os.path.join(system_root, user_path)
    else:
        return user_path


def read_best_threshold(path):
    try:
        df = pd.read_csv(path, sep='\t')
        if 'Best_Threshold' in df.columns:
            return float(df['Best_Threshold'].iloc[0])
    except Exception:
        return np.nan
    return np.nan


def get_model_features(model):
    # XGBoost Booster
    try:
        feats = model.feature_names
        if feats is None:
            return []
        return list(feats)
    except Exception:
        try:
            # sklearn wrapper
            return list(model.get_booster().feature_names)
        except Exception:
            return []


def preprocess_data(df, annotation_column, neg_label, pos_label):
    # remove rows with NA in annotation column
    df = df[~df[annotation_column].isna()].copy()
    # keep only rows with annotation in neg or pos labels
    df = df[df[annotation_column].isin([neg_label, pos_label])].copy()
    df[annotation_column] = pd.Categorical(df[annotation_column], categories=[neg_label, pos_label])
    return df


def do_merge_static(annot_df, protein_df, annotation_column, subset_ids=None):
    """Transpose protein_df (index=protein IDs, cols sample IDs) to samples x features
    and left-join annotation on sample_id. subset_ids are feature IDs to keep (optional)."""
    prot_t = protein_df.T.reset_index()
    prot_t = prot_t.rename(columns={'index': 'sample_id'})
    if annot_df is not None:
        df_merged = pd.merge(prot_t, annot_df, how='left', on='sample_id')
    else:
        df_merged = prot_t
    # clean column names like R did (remove ;.*)
    df_merged.columns = [c.split(';')[0] for c in df_merged.columns]
    if subset_ids is not None and len(subset_ids) > 0:
        keep_cols = ['sample_id', annotation_column] + [x for x in subset_ids if x in df_merged.columns]
        df_merged = df_merged[[c for c in keep_cols if c in df_merged.columns]]
    # convert feature columns to numeric and drop columns that are all NA
    keep_always = ['sample_id'] + ([annotation_column] if annotation_column is not None else [])
    for col in list(df_merged.columns):
        if col in keep_always:
            continue
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        if df_merged[col].isna().all():
            df_merged.drop(columns=[col], inplace=True)
    return df_merged
