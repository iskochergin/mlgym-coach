"""Клиент бэкенда Софы для создания задачи и запуска эксперимента.

Переменные окружения:
  MLGYM_API_URL — базовый URL HTTP API (например http://localhost:8000).
  Если не задан, используется локальный fallback (сохранение в tasks/ + runner).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Optional
from uuid import uuid4

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO_ROOT / "runs"
TASKS_DIR = _REPO_ROOT / "tasks"
TASK_SPECS_DIR = TASKS_DIR / "specs"
TASK_DATA_DIR = TASKS_DIR / "data"
HIDDEN_LABELS_DIR = TASKS_DIR / "hidden_labels"
RUNNER_CONFIGS_DIR = RUNS_DIR / "_dashboard_configs"

METRIC_HIGHER_BETTER = {
    "roc_auc": True,
    "accuracy": True,
    "f1": True,
    "rmse": False,
    "mae": False,
}


@dataclass(frozen=True)
class RunLaunch:
    task_id: str
    run_id: str
    run_dir: Path
    backend: str  # "http" | "local"


def api_base_url() -> Optional[str]:
    url = os.environ.get("MLGYM_API_URL", "").strip().rstrip("/")
    return url or None


def slugify_task_id(description: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")
    if not base:
        base = "custom_task"
    return f"{base[:40]}_{uuid4().hex[:6]}"


def submit_run(
    *,
    description: str,
    metric: str,
    agent: str,
    token_budget: int,
    seeds: int,
    train_csv: BinaryIO,
    test_features_csv: BinaryIO,
    sample_csv: Optional[BinaryIO] = None,
    hidden_labels_csv: Optional[BinaryIO] = None,
    llm_mode: str = "mock",
    model: str = "fake-model",
    max_steps: int = 12,
) -> RunLaunch:
    if api_base_url():
        return _submit_run_http(
            description=description,
            metric=metric,
            agent=agent,
            token_budget=token_budget,
            seeds=seeds,
            train_csv=train_csv,
            test_features_csv=test_features_csv,
            sample_csv=sample_csv,
            hidden_labels_csv=hidden_labels_csv,
            llm_mode=llm_mode,
            model=model,
            max_steps=max_steps,
        )
    return _submit_run_local(
        description=description,
        metric=metric,
        agent=agent,
        token_budget=token_budget,
        seeds=seeds,
        train_csv=train_csv,
        test_features_csv=test_features_csv,
        sample_csv=sample_csv,
        hidden_labels_csv=hidden_labels_csv,
        llm_mode=llm_mode,
        model=model,
        max_steps=max_steps,
    )


def _submit_run_http(**kwargs: Any) -> RunLaunch:
    base = api_base_url()
    assert base is not None

    # Контракт API согласуется с Софой; пока минимальный POST /runs.
    boundary = f"----mlgym-{uuid4().hex}"
    body, content_type = _build_multipart(boundary, kwargs)

    req = urllib.request.Request(
        f"{base}/runs",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend error {exc.code}: {detail}") from exc

    task_id = str(payload["task_id"])
    run_id = str(payload["run_id"])
    run_dir = RUNS_DIR / run_id
    return RunLaunch(task_id=task_id, run_id=run_id, run_dir=run_dir, backend="http")


def _build_multipart(boundary: str, fields: dict[str, Any]) -> tuple[bytes, str]:
    lines: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        lines.append(f"{value}\r\n".encode())

    def add_file(name: str, file_obj: BinaryIO, filename: str) -> None:
        content = file_obj.read()
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        lines.append(b"Content-Type: application/octet-stream\r\n\r\n")
        lines.append(content)
        lines.append(b"\r\n")

    add_field("description", str(fields["description"]))
    add_field("metric", str(fields["metric"]))
    add_field("agent", str(fields["agent"]))
    add_field("token_budget", str(fields["token_budget"]))
    add_field("seeds", str(fields["seeds"]))
    add_field("llm_mode", str(fields["llm_mode"]))
    add_field("model", str(fields["model"]))
    add_field("max_steps", str(fields["max_steps"]))

    add_file("train_csv", fields["train_csv"], "train.csv")
    add_file("test_features_csv", fields["test_features_csv"], "test_features.csv")
    if fields.get("sample_csv") is not None:
        add_file("sample_csv", fields["sample_csv"], "sample.csv")
    if fields.get("hidden_labels_csv") is not None:
        add_file("hidden_labels_csv", fields["hidden_labels_csv"], "hidden_labels.csv")

    lines.append(f"--{boundary}--\r\n".encode())
    content_type = f"multipart/form-data; boundary={boundary}"
    return b"".join(lines), content_type


def _submit_run_local(**kwargs: Any) -> RunLaunch:
    description = str(kwargs["description"])
    metric = str(kwargs["metric"])
    agent = str(kwargs["agent"])
    token_budget = int(kwargs["token_budget"])
    seeds = int(kwargs["seeds"])
    llm_mode = str(kwargs["llm_mode"])
    model = str(kwargs["model"])
    max_steps = int(kwargs["max_steps"])

    task_id = slugify_task_id(description)
    run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = TASK_DATA_DIR / task_id
    data_dir.mkdir(parents=True, exist_ok=True)
    _save_upload(kwargs["train_csv"], data_dir / "train.csv")
    _save_upload(kwargs["test_features_csv"], data_dir / "test_features.csv")
    if kwargs.get("sample_csv") is not None:
        _save_upload(kwargs["sample_csv"], data_dir / "sample.csv")

    hidden_rel = f"tasks/hidden_labels/{task_id}/y_test.csv"
    if kwargs.get("hidden_labels_csv") is not None:
        hidden_path = HIDDEN_LABELS_DIR / task_id / "y_test.csv"
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        _save_upload(kwargs["hidden_labels_csv"], hidden_path)

    spec = {
        "id": task_id,
        "description": description,
        "metric": metric,
        "metric_higher_better": METRIC_HIGHER_BETTER.get(metric, True),
        "train_path": f"tasks/data/{task_id}/train.csv",
        "test_features_path": f"tasks/data/{task_id}/test_features.csv",
        "hidden_labels_path": hidden_rel,
    }
    spec_path = TASK_SPECS_DIR / f"{task_id}.yaml"
    TASK_SPECS_DIR.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    config = {
        "experiment_name": run_id,
        "model": model,
        "env": "fake" if llm_mode == "mock" else "real",
        "agents": [agent],
        "seeds": list(range(seeds)),
        "tasks": [str(spec_path.relative_to(_REPO_ROOT))],
        "token_budget": token_budget,
        "max_steps": max_steps,
        "output_dir": "runs",
        "llm_mode": llm_mode,
    }

    RUNNER_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = RUNNER_CONFIGS_DIR / f"{run_id}.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    log_path = run_dir / "runner.log"
    python_bin = _REPO_ROOT / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    env = os.environ.copy()
    env["MLGYM_LLM"] = llm_mode
    env["MLGYM_RUN_DIR"] = str(run_dir)

    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(  # noqa: S603
            [str(python_bin), "-m", "runner.run", "--config", str(cfg_path)],
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    meta = {
        "task_id": task_id,
        "run_id": run_id,
        "agent": agent,
        "metric": metric,
        "seeds": seeds,
        "token_budget": token_budget,
        "llm_mode": llm_mode,
        "pid": proc.pid,
        "status": "running",
        "config_path": str(cfg_path.relative_to(_REPO_ROOT)),
        "partial_path": "episode.partial.json",
        "final_path": "episode.json",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunLaunch(task_id=task_id, run_id=run_id, run_dir=run_dir, backend="local")


def _save_upload(file_obj: BinaryIO, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_obj.read())


def read_run_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def is_process_running(pid: Optional[int]) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_status(run_dir: Path) -> str:
    meta = read_run_meta(run_dir)
    pid = meta.get("pid")
    partial = run_dir / "episode.partial.json"
    final = run_dir / "episode.json"

    if final.exists():
        return "completed"
    if is_process_running(pid):
        return "running"
    if partial.exists() or list(run_dir.rglob("seed_*.json")):
        return "completed"
    return "failed"
