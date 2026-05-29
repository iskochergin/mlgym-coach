from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from core.types import Task
from runner.run import run_experiment
from tasks.metrics import METRICS
from tasks.task_loader import LoadedTask, load_task_by_id


_REPO_ROOT = Path(__file__).resolve().parent.parent
_UPLOADS_DIR = _REPO_ROOT / "tasks" / "uploads"
_HIDDEN_LABELS_DIR = _REPO_ROOT / "tasks" / "hidden_labels"
_DEFAULT_TARGET_COLUMN = "target"
_METRICS_BY_TASK_TYPE = {
    "classification": {"roc_auc"},
    "regression": {"rmse"},
}


@dataclass(frozen=True)
class CreatedTask:
    task_id: str
    task: Task
    spec_path: Path
    visible_dir: Path
    hidden_labels_path: Path
    sample_path: Path | None


def create_task(
    *,
    description: str,
    metric: str,
    train_path: str | Path,
    test_features_path: str | Path,
    hidden_labels_path: str | Path,
    sample_path: str | Path | None = None,
    target_column: str = _DEFAULT_TARGET_COLUMN,
    task_type: str | None = None,
    task_id: str | None = None,
) -> CreatedTask:
    """Register an uploaded tabular task while keeping test labels hidden."""
    metric = metric.strip()
    if metric not in METRICS:
        known = ", ".join(sorted(METRICS))
        raise ValueError(f"Unknown metric {metric!r}. Supported metrics: {known}")

    train_src = _resolve_existing_file(train_path, "train_path")
    test_src = _resolve_existing_file(test_features_path, "test_features_path")
    hidden_src = _resolve_existing_file(hidden_labels_path, "hidden_labels_path")
    sample_src = _resolve_existing_file(sample_path, "sample_path") if sample_path else None

    train = pd.read_csv(train_src)
    test_features = pd.read_csv(test_src)
    hidden_labels = pd.read_csv(hidden_src)
    _validate_uploaded_frames(
        train=train,
        test_features=test_features,
        hidden_labels=hidden_labels,
        target_column=target_column,
        metric=metric,
        task_type=task_type,
    )

    final_task_type = task_type or _infer_task_type(train[target_column], hidden_labels.iloc[:, 0])
    metric_higher_better = metric == "roc_auc"
    final_task_id = _unique_task_id(task_id or _slugify(description))

    visible_dir = _UPLOADS_DIR / final_task_id
    hidden_dir = _HIDDEN_LABELS_DIR / final_task_id
    visible_dir.mkdir(parents=True, exist_ok=False)
    hidden_dir.mkdir(parents=True, exist_ok=False)

    train_dst = visible_dir / "train.csv"
    test_dst = visible_dir / "test_features.csv"
    hidden_dst = hidden_dir / "y_test.csv"
    sample_dst = visible_dir / "sample.csv" if sample_src else None

    shutil.copyfile(train_src, train_dst)
    shutil.copyfile(test_src, test_dst)
    shutil.copyfile(hidden_src, hidden_dst)
    if sample_src and sample_dst:
        shutil.copyfile(sample_src, sample_dst)

    spec = {
        "id": final_task_id,
        "description": description,
        "metric": metric,
        "metric_higher_better": metric_higher_better,
        "task_type": final_task_type,
        "target_column": target_column,
        "train_path": _repo_relative(train_dst),
        "test_features_path": _repo_relative(test_dst),
        "hidden_labels_path": _repo_relative(hidden_dst),
    }
    if sample_dst:
        spec["sample_path"] = _repo_relative(sample_dst)

    spec_path = hidden_dir / "task.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    loaded = load_task_by_id(final_task_id)
    return CreatedTask(
        task_id=final_task_id,
        task=loaded.task,
        spec_path=spec_path,
        visible_dir=visible_dir,
        hidden_labels_path=hidden_dst,
        sample_path=sample_dst,
    )


def load_uploaded_task(task_id: str) -> LoadedTask:
    return load_task_by_id(task_id)


def run_uploaded_task(
    *,
    task_id: str,
    num_seeds: int = 3,
    agents: list[str] | None = None,
    env: str = "fake",
    model: str = "fake-model",
    output_dir: str | Path = "runs",
    max_steps: int = 6,
    token_budget: int = 4000,
) -> list[Path]:
    if not 1 <= num_seeds <= 10:
        raise ValueError(f"num_seeds must be in range 1..10, got {num_seeds}")
    config: dict[str, Any] = {
        "experiment_name": f"adhoc_{task_id}",
        "model": model,
        "env": env,
        "agents": agents or ["baseline", "scaffold"],
        "seeds": list(range(num_seeds)),
        "task_ids": [task_id],
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "token_budget": token_budget,
    }
    return run_experiment(config)


def _validate_uploaded_frames(
    *,
    train: pd.DataFrame,
    test_features: pd.DataFrame,
    hidden_labels: pd.DataFrame,
    target_column: str,
    metric: str,
    task_type: str | None,
) -> None:
    if target_column not in train.columns:
        raise ValueError(f"Train file must contain target column {target_column!r}")
    if target_column in test_features.columns:
        raise ValueError(f"Test features must not contain target column {target_column!r}")
    if hidden_labels.shape[1] != 1:
        raise ValueError("Hidden labels file must contain exactly one column")
    if len(test_features) != len(hidden_labels):
        raise ValueError(
            "test_features and hidden_labels row counts differ: "
            f"{len(test_features)} != {len(hidden_labels)}"
        )

    inferred_type = _infer_task_type(train[target_column], hidden_labels.iloc[:, 0])
    final_type = task_type or inferred_type
    if final_type not in _METRICS_BY_TASK_TYPE:
        known = ", ".join(sorted(_METRICS_BY_TASK_TYPE))
        raise ValueError(f"Unknown task_type {final_type!r}. Supported types: {known}")
    if task_type and task_type != inferred_type:
        raise ValueError(
            f"task_type={task_type!r} does not match labels inferred as {inferred_type!r}"
        )
    if metric not in _METRICS_BY_TASK_TYPE[final_type]:
        allowed = ", ".join(sorted(_METRICS_BY_TASK_TYPE[final_type]))
        raise ValueError(f"Metric {metric!r} is not valid for {final_type}; use one of: {allowed}")


def _infer_task_type(train_target: pd.Series, hidden_target: pd.Series) -> str:
    values = pd.concat([train_target, hidden_target], ignore_index=True).dropna()
    unique = set(values.unique().tolist())
    if unique <= {0, 1} or unique <= {False, True}:
        return "classification"
    if pd.api.types.is_numeric_dtype(values):
        return "regression"
    raise ValueError(
        "Could not infer task type from labels. Use binary 0/1 labels for roc_auc "
        "or numeric labels for rmse."
    )


def _resolve_existing_file(path: str | Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required")
    path = Path(path)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:48] or "uploaded_task"


def _unique_task_id(base: str) -> str:
    base = _slugify(base)
    while True:
        candidate = f"{base}_{uuid.uuid4().hex[:8]}"
        if (
            not (_UPLOADS_DIR / candidate).exists()
            and not (_HIDDEN_LABELS_DIR / candidate).exists()
        ):
            return candidate


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(_REPO_ROOT))
