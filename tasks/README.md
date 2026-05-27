# Tasks Data

Prepared datasets live under `tasks/data/<task_id>/`:

- `train.csv` contains features plus `target`.
- `test_features.csv` contains only held-out features.

Held-out labels live separately under `tasks/hidden_labels/<task_id>/y_test.csv`.
Runner/grader code may read those labels, but task specs exposed to agents should not.

Current datasets:

- `breast_cancer_roc_auc`: binary classification, intended metric `roc_auc`.
- `diabetes_rmse`: regression, intended metric `rmse`.

Regenerate with:

```bash
python3 tasks/prepare_datasets.py
```
