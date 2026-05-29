from __future__ import annotations

from typing import Protocol

from core.types import EpisodeResult, Task
from env.runner_adapter import build_real_env
from runner.fake_env import FakeEnv, FakeEnvConfig


class RunnerEnv(Protocol):
    def run(self) -> EpisodeResult: ...


def build_env(
    *,
    env_name: str,
    task: Task,
    agent: str,
    seed: int,
    config: FakeEnvConfig,
    hidden_labels_path: str | None = None,
    run_dir: str | None = None,
) -> RunnerEnv:
    """Single replacement point for fake vs real runner environments."""
    if env_name == "fake":
        return FakeEnv(task=task, agent=agent, seed=seed, config=config)
    if env_name == "real":
        # coach comes from FakeEnvConfig, which is a dataclass rather than dict.
        # Default keeps old configs without a coach field working.
        coach = str(getattr(config, "coach", "dummy"))
        return build_real_env(
            task=task,
            agent=agent,
            seed=seed,
            config=config,
            coach=coach,
            hidden_labels_path=hidden_labels_path,
            run_dir=run_dir,
        )
    raise ValueError(f"Unsupported env {env_name!r}. Supported values: fake, real")
