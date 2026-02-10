#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

os.makedirs('f:\\ProteoBoostR\\multigroup\\test_data', exist_ok=True)

n_samples = 120
n_features = 200
classes = ['Control', 'GroupA', 'GroupB']

# assign labels roughly balanced
labels = np.random.choice(classes, size=n_samples, p=[0.33, 0.33, 0.34])
sample_ids = [f'S{i+1:03d}' for i in range(n_samples)]

# create protein matrix: rows = features, cols = sample_ids
np.random.seed(42)
base = np.random.normal(loc=0.0, scale=1.0, size=(n_features, n_samples))
# add class-specific shifts to make them somewhat separable
for i, lab in enumerate(labels):
    if lab == 'GroupA':
        base[0:10, i] += 1.5
    elif lab == 'GroupB':
        base[10:20, i] -= 1.2

features = [f'Prot_{i+1:04d}' for i in range(n_features)]
protein_df = pd.DataFrame(base, index=features, columns=sample_ids)
protein_df.to_csv('f:\\ProteoBoostR\\multigroup\\test_data\\protein.tsv', sep='\t')

annot_df = pd.DataFrame({'sample_id': sample_ids, 'label': labels})
annot_df.to_csv('f:\\ProteoBoostR\\multigroup\\test_data\\annotation.tsv', sep='\t', index=False)

print('Synthetic dataset written to multigroup/test_data')
