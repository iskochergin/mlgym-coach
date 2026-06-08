"""Stage-policy для Env: жёсткий или гибкий порядок переходов между стадиями.

Применяется только в режимах `fixed` и `flexible` (для single_shot /
repeated_single_shot стадии нерелевантны — это one-shot).
"""
from __future__ import annotations

from typing import Optional, Protocol

from core.types import Action, ActionType, Stage


class StagePolicy(Protocol):
    """Решает, можно ли выполнить действие на текущей стадии.

    Возвращает (allowed, replacement):
      - (True,  None)        — действие выполняется как есть;
      - (False, Action(...)) — env подменяет на replacement-nudge.
    """

    def check(self, current_stage: Stage, action: Action) -> tuple[bool, Optional[Action]]: ...


class FlexibleTransitionsPolicy:
    """Текущее поведение: агент сам выбирает следующий шаг, никаких ограничений."""

    name = "flexible"

    def check(self, current_stage: Stage, action: Action) -> tuple[bool, Optional[Action]]:
        return True, None


class FixedTransitionsPolicy:
    """Жёсткий порядок: EDA → BASELINE → IMPROVE → SUBMIT, без возвратов и прыжков.

    Любое действие, нарушающее порядок (например, SUBMIT на EDA), подменяется
    на nudge `Action(PLAN, "следующий ожидаемый шаг: <stage>")`, чтобы агент
    смог переориентироваться. Запретные переходы:
      * SUBMIT раньше IMPROVE;
      * CODE/EDA после SUBMIT;
      * любое действие, явно «прыгающее» назад.
    """

    name = "fixed"
    _ORDER = [Stage.EDA, Stage.BASELINE, Stage.IMPROVE, Stage.SUBMIT]

    def check(self, current_stage: Stage, action: Action) -> tuple[bool, Optional[Action]]:
        cur_idx = self._ORDER.index(current_stage)
        # SUBMIT разрешён только если мы уже на IMPROVE или SUBMIT.
        if action.type == ActionType.SUBMIT and current_stage in (Stage.EDA, Stage.BASELINE):
            return False, Action(
                type=ActionType.PLAN,
                content=(
                    "[FIXED_POLICY] SUBMIT не разрешён на стадии "
                    f"{current_stage.value}; пройди сначала {Stage.BASELINE.value} → "
                    f"{Stage.IMPROVE.value}."
                ),
            )
        # CODE после SUBMIT — попытка «возврата» — не разрешена.
        if current_stage == Stage.SUBMIT and action.type in (ActionType.CODE, ActionType.EDA):
            return False, Action(
                type=ActionType.PLAN,
                content="[FIXED_POLICY] Эпизод уже в стадии submit, возврат назад запрещён.",
            )
        # EDA после улучшения (IMPROVE) — тоже шаг назад.
        if current_stage == Stage.IMPROVE and action.type == ActionType.EDA:
            return False, Action(
                type=ActionType.PLAN,
                content="[FIXED_POLICY] EDA уже пройдена, не возвращайся — пиши CODE или RUN.",
            )
        return True, None


def resolve_stage_policy(name: Optional[str]) -> StagePolicy:
    """Имя → StagePolicy. Дефолт — flexible."""
    if name is None or name == "flexible":
        return FlexibleTransitionsPolicy()
    if name == "fixed":
        return FixedTransitionsPolicy()
    raise ValueError(f"unknown stage_policy: {name!r}; expected 'flexible' or 'fixed'")
