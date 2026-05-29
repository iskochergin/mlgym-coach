# Runner

Run the current fake-env smoke experiment:

```bash
python3 -m runner.run --config runner/configs/fake_smoke.yaml
```

Run through the current `env.gym.Env` adapter:

```bash
python3 -m runner.run --config runner/configs/real_smoke.yaml
```

Register an uploaded task from UI/backend code:

```python
from runner.api import create_task, run_uploaded_task

created = create_task(
    description="Binary churn classification",
    metric="roc_auc",
    train_path="/path/to/train.csv",
    test_features_path="/path/to/test_features.csv",
    hidden_labels_path="/path/to/y_test.csv",
)
run_uploaded_task(task_id=created.task_id, num_seeds=3)
```

The upload API stores `train.csv`, `test_features.csv`, and optional
`sample.csv` under `tasks/uploads/<task_id>/`. Hidden labels are stored only at
`tasks/hidden_labels/<task_id>/y_test.csv`, matching the grader/env convention.
Ad-hoc runner configs can use `task_ids` plus `num_seeds` instead of pre-baked
task spec paths; see `runner/configs/adhoc_example.yaml`.

Aggregate saved `EpisodeResult` files into `reports/`:

```bash
python3 -m runner.aggregate \
  --input runs/fake_smoke \
  --output reports/fake_smoke_summary
```

Run the end-to-end smoke check in a temporary directory:

```bash
python3 -m runner.smoke_check
```

The runner supports `env: real` through `env.runner_adapter.build_real_env` and
keeps `env: fake` for deterministic runner tests. Keep the `EpisodeResult` JSON
format unchanged; the aggregator only depends on that frozen contract.

Each episode gets its own live directory:
`runs/<experiment_name>/<task_id>/seed<N>/<agent>/episode.json`. The real env
writes `episode.partial.json` there during execution and atomically replaces it
with the final `episode.json` at the end.
