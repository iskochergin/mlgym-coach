# Tasks Data

Prepared datasets live under `tasks/data/<task_id>/`:

- `train.csv` contains features plus `target`.
- `test_features.csv` contains only held-out features.

Held-out labels live separately under `tasks/hidden_labels/<task_id>/y_test.csv`.
Runner/grader code may read those labels, but task specs exposed to agents should not.

Uploaded ad-hoc tasks follow the same split:

- visible files go to `tasks/uploads/<task_id>/`
- hidden labels go to `tasks/hidden_labels/<task_id>/y_test.csv`

Use `runner.api.create_task(...)` to validate and register uploaded CSV files.
Private runner metadata with the hidden-label path is stored beside
`y_test.csv`, not in the visible upload directory.

Current datasets:

- `breast_cancer_roc_auc`: binary classification, intended metric `roc_auc`.
- `diabetes_rmse`: regression, intended metric `rmse`.

Regenerate with:

```bash
python3 tasks/prepare_datasets.py
```
