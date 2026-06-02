from __future__ import annotations

from typing import Callable, Optional, Protocol

from core.types import EpisodeResult, Task
from runner.fake_env import FakeEnv, FakeEnvConfig
from runner.real_env import RealGymRunnerEnv


class RunnerEnv(Protocol):
    def run(self, on_step: Optional[Callable[[EpisodeResult], None]] = None) -> EpisodeResult: ...


def build_env(
    *,
    env_name: str,
    task: Task,
    agent: str,
    seed: int,
    config: FakeEnvConfig,
) -> RunnerEnv:
    """Single replacement point for fake vs real runner environments."""
    if env_name == "fake":
        return FakeEnv(task=task, agent=agent, seed=seed, config=config)
    if env_name == "real":
        return RealGymRunnerEnv(task=task, agent=agent, seed=seed, config=config)
    raise ValueError(f"Unsupported env {env_name!r}. Supported values: fake, real")
