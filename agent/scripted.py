"""ScriptedAgent — проигрывает заранее заданную последовательность Action.

Никакого LLM. Нужен, чтобы тестировать петлю Env без модели.
"""
from __future__ import annotations

from typing import Iterable

from core.types import Action, Observation


class ScriptedAgent:
    def __init__(self, actions: Iterable[Action]) -> None:
        self._actions: list[Action] = list(actions)
        self._idx: int = 0

    def act(self, obs: Observation) -> Action:
        if self._idx >= len(self._actions):
            raise RuntimeError("ScriptedAgent: actions exhausted")
        action = self._actions[self._idx]
        self._idx += 1
        return action

    @property
    def remaining(self) -> int:
        return len(self._actions) - self._idx
