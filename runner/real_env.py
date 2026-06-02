from __future__ import annotations
from typing import Any, Callable, Optional

from core.types import Action, ActionType, EpisodeResult, Task
from env.dummy_coach import DummyCoach
from env.executor import reset_executor
from env.gym import Env
from runner.fake_env import FakeEnvConfig


class RealGymRunnerEnv:
    """Adapter from runner's one-shot interface to env.gym.Env's step loop."""

    def __init__(self, task: Task, agent: str, seed: int, config: FakeEnvConfig) -> None:
        self.task = task
        self.agent = agent
        self.seed = seed
        self.config = config

    def run(self, on_step: Optional[Callable[[EpisodeResult], None]] = None) -> EpisodeResult:
        reset_executor()
        env = Env(
            task=self.task,
            coach=DummyCoach(),
            token_budget=self.config.token_budget,
            seed=self.seed,
            agent_name=self.agent,
        )

        obs = env.reset()
        if on_step:
            on_step(env.result())

        for action in _scripted_actions(self.agent, self.task.metric)[: self.config.max_steps]:
            obs = env.step(action)
            if on_step:
                res = env.result()
                res.config.update(self._episode_config())
                on_step(res)
            if action.type == ActionType.SUBMIT or obs.tokens_left <= 0:
                break

        result = env.result()
        result.config.update(self._episode_config())
        return result

    def _episode_config(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "agent_kind": self.agent,
            "env": self.config.env_name,
            "max_steps": self.config.max_steps,
            "budget_tokens": self.config.token_budget,
            "metric": self.task.metric,
            "metric_higher_better": self.task.metric_higher_better,
        }


def _scripted_actions(agent: str, metric: str) -> list[Action]:
    if agent == "scaffold":
        return [
            Action(
                type=ActionType.PLAN,
                content=f"Plan validation, EDA, baseline, improvement, and submit for {metric}.",
            ),
            Action(
                type=ActionType.EDA,
                content="Inspect schema, target distribution, missingness, and leakage risks.",
            ),
            Action(
                type=ActionType.CODE,
                content="Build a baseline model with a fixed validation split.",
            ),
            Action(type=ActionType.RUN, content="run baseline solution"),
            Action(
                type=ActionType.CODE,
                content="Improve preprocessing or model family after validation feedback.",
            ),
            Action(type=ActionType.SUBMIT, content="submit final predictions"),
        ]

    return [
        Action(
            type=ActionType.PLAN,
            content=f"Quickly build a baseline, try one improvement, then submit for {metric}.",
        ),
        Action(
            type=ActionType.CODE,
            content="Build a simple baseline model with a fixed validation split.",
        ),
        Action(type=ActionType.RUN, content="run baseline solution"),
        Action(
            type=ActionType.CODE,
            content="Try one stronger model variant.",
        ),
        Action(type=ActionType.RUN, content="run improved solution"),
        Action(type=ActionType.SUBMIT, content="submit final predictions"),
    ]
