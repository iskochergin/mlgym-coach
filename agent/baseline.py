"""BaselineAgent — ReAct-агент поверх LLM. Одно действие за вызов act().

Собирает компактный промпт из Observation (задача, метрика, стадия, КРАТКАЯ
сводка истории, последний результат, val_score, бюджет, подсказки), зовёт LLM
и парсит ответ через env.parser.parse_action.
"""
from __future__ import annotations

from typing import Optional

from agent.llm_client import build_client
from core.types import Action, ActionType, Observation, Step
from env.parser import parse_action

_SYSTEM = """\
Ты — ML-инженер, решаешь табличную ML-задачу по шагам (ReAct).
За один ход — РОВНО ОДНО действие в формате:

[ACTION:<type>]
<содержимое>
[/ACTION]

где <type> ∈ {plan, eda, code, run, submit}.
- plan  — короткое рассуждение/план.
- eda   — заметки об исследовании данных.
- code  — ПОЛНЫЙ python-скрипт решения. Он обучается на CSV из переменной
          окружения TRAIN_PATH, делает внутренний валид-сплит и печатает
          строку ровно вида VAL_SCORE=<float>. В режиме сабмита (переменная
          PREDICT=1) дополнительно читает TEST_PATH и пишет predictions.csv
          с одной колонкой предсказаний. y_test тебе недоступен.
- run    — исполнить текущий код и получить валид-метрику.
- submit — финальный прогон с записью предсказаний на тест.
Типичный путь: plan → code → run → (улучшить code → run) → submit.
"""


def _summarize_history(history: list[Step], limit: int = 6) -> str:
    if not history:
        return "(история пуста)"
    tail = history[-limit:]
    lines = []
    for s in tail:
        score = f" val={s.val_score:.4f}" if s.val_score is not None else ""
        result = s.result.replace("\n", " ")
        if len(result) > 120:
            result = result[:120] + "…"
        lines.append(f"#{s.idx} [{s.stage.value}/{s.action.type.value}]{score} → {result}")
    return "\n".join(lines)


class BaselineAgent:
    def __init__(self, client=None) -> None:
        self.client = client if client is not None else build_client()

    def act(self, obs: Observation) -> Action:
        prompt = self._build_prompt(obs)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        raw = self.client.complete(messages, max_tokens=2000)
        return parse_action(raw)

    def _build_prompt(self, obs: Observation) -> str:
        task = obs.task
        parts = [
            f"Задача: {task.description}",
            f"Метрика: {task.metric} ({'выше лучше' if task.metric_higher_better else 'ниже лучше'})",
            f"Текущая стадия: {obs.stage.value}",
            f"Бюджет токенов: {obs.tokens_left}",
            "",
            "Сводка истории:",
            _summarize_history(obs.history),
        ]
        if obs.last_result:
            last = obs.last_result if len(obs.last_result) <= 400 else obs.last_result[:400] + "…"
            parts += ["", f"Последний результат среды:\n{last}"]
        if obs.val_score is not None:
            parts.append(f"Текущий val_score: {obs.val_score:.4f}")
        if obs.hints:
            parts.append("")
            parts.append("Подсказки коуча:")
            for h in obs.hints:
                parts.append(f"  - [L{h.level}] {h.text}")

        # Жёсткий nudge по истории: говорим прямо, какой ход ОБЯЗАТЕЛЬНЫЙ дальше,
        # чтобы модель не залипала в бесконечном planning.
        nudge = _next_step_hint(obs)
        if nudge:
            parts += ["", nudge]
        parts += [
            "",
            "Выдай ровно одно следующее действие в требуемом формате [ACTION:type]…[/ACTION].",
        ]
        return "\n".join(parts)


def _next_step_hint(obs: Observation) -> str:
    """По истории сказать модели, что от неё ждут СЛЕДУЮЩИМ шагом."""
    types = [s.action.type for s in obs.history]
    has_code = ActionType.CODE in types
    has_run_with_score = any(
        s.action.type == ActionType.RUN and s.val_score is not None for s in obs.history
    )
    has_run = ActionType.RUN in types

    if not has_code:
        if len(types) >= 1:
            return (
                "СЛЕДУЮЩИЙ ход ОБЯЗАН быть [ACTION:code] с ПОЛНЫМ рабочим python-скриптом "
                "(чтение TRAIN_PATH, обучение, печать VAL_SCORE=<float>, поддержка PREDICT=1 "
                "для записи predictions.csv по TEST_PATH). Без code дальше идти нельзя."
            )
        return ""  # совсем пусто — пусть начнёт с plan/eda естественно
    if has_code and not has_run:
        return "У тебя уже есть код. СЛЕДУЮЩИЙ ход — [ACTION:run], чтобы получить VAL_SCORE."
    if has_run_with_score:
        return (
            "Уже есть валидационный скор. Либо улучши решение через [ACTION:code], либо "
            "финализируй через [ACTION:submit]."
        )
    return ""
