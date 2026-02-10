Python scripts port of ProteoBoostR core functionality.

Files:
- `utils.py`: helper functions for preprocessing and merging
- `train.py`: train model with Bayesian optimization and save model/params
- `evaluate.py`: evaluate a saved model on labeled test data
- `apply_model.py`: apply saved model to new datasets (threshold banding)
- `requirements.txt`: dependencies

Basic usage examples:

Train:
```
python py_scripts/train.py --annotation GBM_testcase/Werner_annot.tsv \
    --protein GBM_testcase/Werner_data.tsv --annotcol subtypes --neg subtype2+3+4 --pos subtype1 \
    --output outputs --testsize 0.3 --seed 123
```

Evaluate:
```
python py_scripts/evaluate.py --model outputs/xgb_model_2025...json --annotation GBM_testcase/Werner_annot.tsv \
    --protein GBM_testcase/Werner_data.tsv --annotcol subtypes --neg subtype2+3+4 --pos subtype1 --output outputs
```

Apply model to new cohort:
```
python py_scripts/apply_model.py --model outputs/xgb_model_2025...json --protein CPTAC_data.tsv \
    --annotation CPTAC_annot.tsv --annotcol subtypes --neg subtype2+3+4 --pos subtype1 --evaltsv outputs/evaluation_results_...tsv \
    --band 0.1 --output outputs
```

Unified CLI
```
python py_scripts/cli.py train --annotation GBM_testcase/Werner_annot.tsv \
    --protein GBM_testcase/Werner_data.tsv --annotcol subtypes --neg subtype2+3+4 --pos subtype1 \
    --output outputs --testsize 0.3 --seed 123
```

Evaluate using unified CLI:
```
python py_scripts/cli.py evaluate --model outputs/xgb_model_2025...json --annotation GBM_testcase/Werner_annot.tsv \
    --protein GBM_testcase/Werner_data.tsv --annotcol subtypes --neg subtype2+3+4 --pos subtype1 --output outputs
```

Apply model using unified CLI:
```
python py_scripts/cli.py apply --model outputs/xgb_model_2025...json --protein CPTAC_data.tsv \
    --annotation CPTAC_annot.tsv --annotcol subtypes --neg subtype2+3+4 --pos subtype1 --evaltsv outputs/evaluation_results_...tsv \
    --band 0.1 --output outputs
```
