# TODO v1 stub — заменить на реальную песочницу (subprocess + timeout + train.csv) в шаге 5b.
"""Заглушка-исполнитель. Реально код агента не запускает.

Возвращает фейковый stdout ("VAL_SCORE=...") и растущий по вызовам скор, чтобы Env
мог замкнуть петлю и проверить интерфейс без настоящей песочницы.

В шаге 5b — заменить на настоящего раннера: subprocess с таймаутом, ограниченным
доступом к ФС (только train.csv задачи), захватом stdout/stderr и парсингом метрики.
"""
from __future__ import annotations

from typing import Optional

# Кривая «как будто агент улучшается» — последовательность валидационных скоров,
# которую стаб выдаёт на последовательные RUN-вызовы.
_VAL_CURVE: list[float] = [0.75, 0.79, 0.82, 0.84, 0.86]
_SUBMIT_SCORE: float = 0.835

_call_count: int = 0


def reset_executor() -> None:
    """Сбросить состояние стаба (для повторных эпизодов в одном процессе)."""
    global _call_count
    _call_count = 0


def run_solution(code: str, task, mode: str = "run") -> tuple[str, Optional[float]]:
    """Псевдо-исполнение кода.

    mode="run"    → возвращает следующий val-скор из _VAL_CURVE.
    mode="submit" → возвращает финальный test-скор.
    """
    global _call_count
    if mode == "submit":
        return (
            f"submission accepted. test {task.metric}={_SUBMIT_SCORE:.3f}",
            _SUBMIT_SCORE,
        )
    idx = min(_call_count, len(_VAL_CURVE) - 1)
    score = _VAL_CURVE[idx]
    _call_count += 1
    return f"VAL_SCORE={score:.3f}", score
