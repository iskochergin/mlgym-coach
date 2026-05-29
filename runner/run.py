from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.types import EpisodeResult
from runner.env_factory import build_env
from runner.fake_env import FakeEnvConfig
from tasks.task_loader import LoadedTask, load_task, load_task_by_id


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    written = run_experiment(config)
    for path in written:
        print(path.relative_to(_REPO_ROOT))


def run_experiment(config: dict[str, Any]) -> list[Path]:
    experiment_name = str(config["experiment_name"])
    output_root = _resolve_repo_path(config.get("output_dir", "runs")) / experiment_name
    output_root.mkdir(parents=True, exist_ok=True)

    env_name = str(config.get("env", "fake"))

    loaded_tasks = _load_tasks(config)
    agents = [str(agent) for agent in config.get("agents", ["baseline", "scaffold"])]
    seeds = _load_seeds(config)
    fake_config = FakeEnvConfig(
        model=str(config.get("model", "fake-model")),
        token_budget=int(config.get("token_budget", 4000)),
        max_steps=int(config.get("max_steps", 6)),
        env_name=env_name,
        coach=str(config.get("coach", "dummy")),
    )

    written: list[Path] = []
    for loaded_task in loaded_tasks:
        for agent in agents:
            for seed in seeds:
                result = _run_one(loaded_task, agent, seed, env_name, fake_config)
                path = output_root / loaded_task.task.id / agent / f"seed_{seed}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result.to_json(), encoding="utf-8")
                written.append(path)
    return written


def _run_one(
    loaded_task: LoadedTask,
    agent: str,
    seed: int,
    env_name: str,
    config: FakeEnvConfig,
) -> EpisodeResult:
    env = build_env(
        env_name=env_name,
        task=loaded_task.task,
        agent=agent,
        seed=seed,
        config=config,
    )
    return env.run()


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Runner config must be a mapping: {path}")
    required = ["experiment_name"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required runner config keys: {missing}")
    if "tasks" not in raw and "task_ids" not in raw:
        raise ValueError("Runner config must contain either 'tasks' or 'task_ids'")
    return raw


def _load_tasks(config: dict[str, Any]) -> list[LoadedTask]:
    tasks = [load_task(path) for path in config.get("tasks", [])]
    tasks.extend(load_task_by_id(task_id) for task_id in config.get("task_ids", []))
    if not tasks:
        raise ValueError("No tasks configured; provide 'tasks' or 'task_ids'")
    return tasks


def _load_seeds(config: dict[str, Any]) -> list[int]:
    if "seeds" in config:
        seeds = [int(seed) for seed in config["seeds"]]
    else:
        num_seeds = int(config.get("num_seeds", 1))
        if not 1 <= num_seeds <= 10:
            raise ValueError(f"num_seeds must be in range 1..10, got {num_seeds}")
        seeds = list(range(num_seeds))
    if len(seeds) > 10:
        raise ValueError(f"At most 10 seeds are allowed, got {len(seeds)}")
    return seeds


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mlgym-coach experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "runner" / "configs" / "fake_smoke.yaml",
        help="Path to YAML runner config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
