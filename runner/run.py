from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runner.fake_env import FakeEnv, FakeEnvConfig
from tasks.task_loader import LoadedTask, load_task


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
    if env_name != "fake":
        raise ValueError(f"Unsupported env {env_name!r}; only 'fake' is wired for now")

    loaded_tasks = [load_task(path) for path in config["tasks"]]
    agents = [str(agent) for agent in config.get("agents", ["baseline", "scaffold"])]
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    fake_config = FakeEnvConfig(
        model=str(config.get("model", "fake-model")),
        token_budget=int(config.get("token_budget", 4000)),
        max_steps=int(config.get("max_steps", 6)),
        env_name=env_name,
    )

    written: list[Path] = []
    for loaded_task in loaded_tasks:
        for agent in agents:
            for seed in seeds:
                result = _run_one(loaded_task, agent, seed, fake_config)
                path = output_root / loaded_task.task.id / agent / f"seed_{seed}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result.to_json(), encoding="utf-8")
                written.append(path)
    return written


def _run_one(loaded_task: LoadedTask, agent: str, seed: int, config: FakeEnvConfig):
    env = FakeEnv(task=loaded_task.task, agent=agent, seed=seed, config=config)
    return env.run()


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Runner config must be a mapping: {path}")
    required = ["experiment_name", "tasks"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required runner config keys: {missing}")
    return raw


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
