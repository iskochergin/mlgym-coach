from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.types import EpisodeResult
from runner.aggregate import aggregate
from runner.run import run_experiment
from tasks.task_loader import load_task


CONFIG_PATH = _REPO_ROOT / "runner" / "configs" / "fake_smoke.yaml"


def main() -> None:
    config = _load_config()
    work_dir = Path(tempfile.mkdtemp(prefix="mlgym-coach-smoke-"))
    try:
        config["experiment_name"] = "smoke_check"
        config["output_dir"] = str(work_dir / "runs")
        written = run_experiment(config)

        expected = len(config["tasks"]) * len(config["agents"]) * len(config["seeds"])
        _assert(len(written) == expected, f"expected {expected} episodes, got {len(written)}")
        _check_episode_json(written)
        _check_task_files(config["tasks"])

        input_dir = work_dir / "runs" / "smoke_check"
        csv_path, md_path = aggregate(input_dir, work_dir / "reports" / "smoke_check_summary")
        _assert(csv_path.exists(), f"missing csv report: {csv_path}")
        _assert(md_path.exists(), f"missing markdown report: {md_path}")
        _assert("baseline" in md_path.read_text(encoding="utf-8"), "markdown report lacks baseline")
        _assert("scaffold" in md_path.read_text(encoding="utf-8"), "markdown report lacks scaffold")

        print(f"smoke check ok: {len(written)} episodes, reports written")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _load_config() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Runner config must be a mapping: {CONFIG_PATH}")
    return raw


def _check_episode_json(paths: list[Path]) -> None:
    for path in paths:
        episode = EpisodeResult.from_json(path.read_text(encoding="utf-8"))
        _assert(episode.task_id, f"missing task_id in {path}")
        _assert(episode.agent in {"baseline", "scaffold"}, f"bad agent in {path}")
        _assert(episode.final_test_score is not None, f"missing final score in {path}")
        _assert(episode.steps, f"missing steps in {path}")
        _assert(episode.total_tokens > 0, f"missing tokens in {path}")


def _check_task_files(task_specs: list[str]) -> None:
    for spec in task_specs:
        loaded = load_task(spec)
        test_features = pd.read_csv(loaded.task.test_features_path)
        hidden_labels = pd.read_csv(loaded.hidden_labels_path)
        _assert("target" not in test_features.columns, f"target leaked into {loaded.task.id}")
        _assert(
            len(test_features) == len(hidden_labels),
            f"hidden label row mismatch for {loaded.task.id}",
        )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
