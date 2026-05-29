"""Парсер ответа LLM в core.types.Action.

Конвенция формата (просим LLM отвечать так):

    [ACTION:<type>]
    <content>
    [/ACTION]

где <type> — одно из значений ActionType (plan/eda/code/run/submit), регистр
не важен. Внутри тега — содержимое: текст плана/EDA или код решения.

Фолбэки (НИКОГДА не падаем):
  - закрывающий тег потерян → берём всё после открывающего;
  - тега нет, но есть ```...``` блок → считаем это CODE;
  - совсем ничего не распарсили → PLAN с сырым текстом.
"""
from __future__ import annotations

import re

from core.types import Action, ActionType

_ACTION_RE = re.compile(
    r"\[ACTION:\s*(?P<type>\w+)\s*\](?P<body>.*?)(?:\[/ACTION\]|$)",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"```(?:\w+)?\n(?P<code>.*?)```", re.DOTALL)


def _coerce_type(raw: str) -> ActionType | None:
    raw = raw.strip().lower()
    try:
        return ActionType(raw)
    except ValueError:
        return None


def parse_action(text: str) -> Action:
    if not text or not text.strip():
        return Action(type=ActionType.PLAN, content="")

    match = _ACTION_RE.search(text)
    if match:
        action_type = _coerce_type(match.group("type"))
        body = match.group("body").strip()
        if action_type is not None:
            return Action(type=action_type, content=body)

    fence = _FENCE_RE.search(text)
    if fence:
        return Action(type=ActionType.CODE, content=fence.group("code").strip())

    return Action(type=ActionType.PLAN, content=text.strip())
