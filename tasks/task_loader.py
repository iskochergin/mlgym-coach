from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.types import Task
from tasks.metrics import MetricFn, get_metric


_REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = _REPO_ROOT / "tasks" / "uploads"
HIDDEN_LABELS_DIR = _REPO_ROOT / "tasks" / "hidden_labels"
SPECS_DIR = _REPO_ROOT / "tasks" / "specs"


@dataclass(frozen=True)
class LoadedTask:
    task: Task
    hidden_labels_path: Path
    metric_fn: MetricFn


def load_task(path: str | Path) -> LoadedTask:
    spec_path = _resolve_repo_path(path)
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task spec must be a mapping: {spec_path}")

    task_kwargs = _task_kwargs(raw, spec_path)
    hidden_labels_path = _required_path(raw, "hidden_labels_path", spec_path)

    task = Task(**task_kwargs)
    metric_fn = get_metric(task.metric)
    return LoadedTask(
        task=task,
        hidden_labels_path=hidden_labels_path,
        metric_fn=metric_fn,
    )


def load_task_by_id(task_id: str) -> LoadedTask:
    candidates = [
        HIDDEN_LABELS_DIR / task_id / "task.yaml",
        UPLOADS_DIR / task_id / "task.yaml",
        SPECS_DIR / f"{task_id}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return load_task(path)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Task id {task_id!r} not found. Searched: {searched}")


def _task_kwargs(raw: dict[str, Any], spec_path: Path) -> dict[str, Any]:
    keys = [
        "id",
        "description",
        "metric",
        "metric_higher_better",
        "train_path",
        "test_features_path",
    ]
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError(f"Missing required Task fields in {spec_path}: {missing}")

    train_path = _required_path(raw, "train_path", spec_path)
    test_features_path = _required_path(raw, "test_features_path", spec_path)

    return {
        "id": str(raw["id"]),
        "description": str(raw["description"]),
        "metric": str(raw["metric"]),
        "metric_higher_better": bool(raw["metric_higher_better"]),
        "train_path": str(train_path),
        "test_features_path": str(test_features_path),
    }


def _required_path(raw: dict[str, Any], key: str, spec_path: Path) -> Path:
    if key not in raw:
        raise ValueError(f"Missing required field {key!r} in {spec_path}")
    path = _resolve_repo_path(raw[key])
    if not path.exists():
        raise FileNotFoundError(f"{key} does not exist for {spec_path}: {path}")
    return path


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()
