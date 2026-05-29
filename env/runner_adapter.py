"""Прослойка между раннером Софы и средой env.gym.Env.

runner/env_factory.build_env ожидает объект с методом .run() -> EpisodeResult
(см. протокол RunnerEnv). Наш Env работает в стиле reset/step/result, поэтому
здесь оборачиваем «Env + агент + петля» в один .run().

Зоны не нарушаем: runner/ не трогаем. Чтобы переключить раннер на настоящую
среду, Софе достаточно в runner/env_factory.py для ветки env_name == "real"
вернуть build_real_env(...) отсюда (одна строка — см. заметку в конце задачи).
"""
from __future__ import annotations

from typing import Optional

from agent.baseline import BaselineAgent
from core.types import ActionType, EpisodeResult, Task
from env.executor import reset_executor
from env.gym import Env


class RealEnv:
    """RunnerEnv-совместимая обёртка: гоняет ReAct-петлю до SUBMIT/лимитов."""

    def __init__(
        self,
        *,
        task: Task,
        agent: str,
        seed: int,
        config,
        coach=None,
        agent_obj=None,
    ) -> None:
        self.task = task
        self.agent = agent
        self.seed = seed
        self.config = config
        self._coach = coach
        self._agent = agent_obj if agent_obj is not None else BaselineAgent()
        self._max_steps = int(getattr(config, "max_steps", 12))
        self._token_budget = int(getattr(config, "token_budget", 50_000))

    def run(self) -> EpisodeResult:
        reset_executor()
        env = Env(
            task=self.task,
            coach=self._coach,
            token_budget=self._token_budget,
            seed=self.seed,
            agent_name=self.agent,
        )
        obs = env.reset()
        for _ in range(self._max_steps):
            action = self._agent.act(obs)
            obs = env.step(action)
            if action.type == ActionType.SUBMIT:
                break
            if obs.tokens_left <= 0:
                break
        return env.result()


def build_real_env(*, task: Task, agent: str, seed: int, config) -> RealEnv:
    """Фабрика для runner/env_factory (ветка env_name == 'real')."""
    return RealEnv(task=task, agent=agent, seed=seed, config=config)
