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
) -> RunnerEnv:
    """Single replacement point for fake vs real runner environments."""
    if env_name == "fake":
        return FakeEnv(task=task, agent=agent, seed=seed, config=config)
    if env_name == "real":
        # coach: "dummy" (baseline) | "real" (scaffold). Берём из конфига, дефолт
        # "dummy" — старые конфиги без поля coach не ломаются. getattr (а не
        # config.get), т.к. config — это FakeEnvConfig dataclass, а не dict.
        coach = str(getattr(config, "coach", "dummy"))
        return build_real_env(task=task, agent=agent, seed=seed, config=config, coach=coach)
    raise ValueError(f"Unsupported env {env_name!r}. Supported values: fake, real")
