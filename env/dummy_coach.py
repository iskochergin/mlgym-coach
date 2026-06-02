"""Заглушка коуча — Env работает без зависимости от настоящего coach.Coach (Егор).

Всегда возвращает (0.0, []) — нулевое покрытие чек-листа, ноль подсказок.
Когда Егор подготовит настоящего Coach, его можно подставить в Env вместо этого
DummyCoach без других изменений: интерфейс assess(obs) -> (float, list[Hint]) общий.
"""
from __future__ import annotations

from core.types import Hint, Observation


class DummyCoach:
    """Молчаливый коуч-заглушка. Покрытие 0, подсказок нет."""

    def assess(self, obs: Observation) -> tuple[float, dict[str, float], list[Hint]]:
        return 0.0, {}, []
