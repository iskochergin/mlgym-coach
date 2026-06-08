"""BaselineAgent — ReAct-агент поверх LLM. Одно действие за вызов act().

Собирает компактный промпт из Observation (задача, метрика, стадия, КРАТКАЯ
сводка истории, последний результат, val_score, бюджет, подсказки), зовёт LLM
и парсит ответ через env.parser.parse_action.
"""
from __future__ import annotations

from typing import Optional  # noqa: F401  (используется в аннотациях ниже)

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

ВАЖНО про конвенцию данных. ВСЕГДА:
  * train.csv содержит и фичи, и колонку с таргетом ВСЕГДА с именем `target`
    (НЕ "Survived", не "class", не "income", не "y" — буквально `target`).
  * test_features.csv содержит ТОЛЬКО фичи (без target).
Если решил использовать df["Survived"]/df["class"] и т.п. — ты ошибся,
бери df["target"].

ВАЖНО про окружение и API:
  * Доступны только `pandas`, `numpy`, `scikit-learn`. НЕ используй lightgbm,
    xgboost, catboost — даже если знаешь API, у них здесь другая версия и
    `verbose_eval`/`silent` etc уже сняты. Для бустинга — `sklearn.ensemble.
    HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`
    (быстрые, без зависимостей).
  * `OneHotEncoder`: в sklearn ≥1.2 параметр `sparse` УДАЛЁН, передавай
    `sparse_output=False`. Ещё проще — `pd.get_dummies(df, dummy_na=True)`.
  * `mean_squared_error(..., squared=False)` тоже снят в 1.4 — для RMSE
    делай `np.sqrt(mean_squared_error(...))` или импортируй
    `root_mean_squared_error`.
  * Данные могут содержать строки/категории (sex, embarked, workclass) и NaN.
    `pd.get_dummies(df, dummy_na=True)` + `df.fillna(df.median(numeric_only=True))`
    решают почти всё.
  * test_features.csv должен пройти ТУ ЖЕ обработку, что и train: после
    get_dummies используй `test = test.reindex(columns=train_cols, fill_value=0)`.

Цикл plan→plan→plan без code НЕ продвигает решение. Если нет VAL_SCORE — пиши/чини code.
ПРИОРИТЕТ: сначала простое рабочее решение (RandomForest или HistGradientBoosting
со стандартными параметрами + get_dummies), потом улучшения.
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
        # Реальный токен-usage последнего вызова LLM (None у MockLLM/при ошибке).
        # Env.step(tokens_used=last_tokens) подменяет эвристику фактическим значением.
        self.last_tokens: Optional[int] = None

    def act(self, obs: Observation) -> Action:
        prompt = self._build_prompt(obs)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        raw = self.client.complete(messages, max_tokens=2000)
        self.last_tokens = getattr(self.client, "last_usage_total", None)
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
    if not obs.history:
        return ""
    last = obs.history[-1]
    types = [s.action.type for s in obs.history]
    has_code = ActionType.CODE in types
    has_run_with_score = any(
        s.action.type == ActionType.RUN and s.val_score is not None for s in obs.history
    )
    last_run = next((s for s in reversed(obs.history) if s.action.type == ActionType.RUN), None)
    last_run_ok = last_run is not None and last_run.val_score is not None

    # 0) Plan-loop сигнал: 2 PLAN подряд в хвосте → СТРОГИЙ запрет третьего.
    last_two = [s.action.type for s in obs.history[-2:]]
    if len(last_two) == 2 and all(t == ActionType.PLAN for t in last_two):
        return (
            "⚠️ Ты уже выдал 2 [ACTION:plan] подряд. ЗАПРЕЩЕНО ещё один plan. "
            "СЛЕДУЮЩИЙ ход — [ACTION:code] (если решения ещё нет) или "
            "[ACTION:run] (если код уже принят). Никаких \"давайте подумаем\"."
        )

    # 1) если буквально только что был CODE — следующий ход ОБЯЗАН быть RUN.
    if last.action.type == ActionType.CODE:
        return "Ты только что прислал код. СЛЕДУЮЩИЙ ход — [ACTION:run]. Не плани, не правь — запускай."

    # 2) был RUN с ошибкой → следующий ход — поправить CODE.
    if last.action.type == ActionType.RUN and last.val_score is None:
        return (
            "Последний RUN упал. СЛЕДУЮЩИЙ ход — [ACTION:code] с исправлением "
            "(скорее всего категории/NaN, см. инструкции в системе)."
        )

    if not has_code:
        return (
            "СЛЕДУЮЩИЙ ход ОБЯЗАН быть [ACTION:code] с ПОЛНЫМ рабочим python-скриптом "
            "(чтение TRAIN_PATH, обучение, печать VAL_SCORE=<float>, поддержка PREDICT=1 "
            "для записи predictions.csv по TEST_PATH). Без code дальше идти нельзя."
        )

    # 3) код был, но без успешного RUN — толкаем в RUN.
    if has_code and not last_run_ok:
        return "У тебя есть код, но ни одного успешного RUN. СЛЕДУЮЩИЙ ход — [ACTION:run]."

    # 4) есть валид-скор → либо улучшать code, либо submit.
    if has_run_with_score:
        return (
            "Уже есть валидационный скор. Если есть идея улучшения — [ACTION:code], "
            "иначе [ACTION:submit] и финализируем эпизод."
        )
    return ""
