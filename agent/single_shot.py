"""SingleShotAgent — один LLM-вызов на весь эпизод, никакого env feedback.

Жизненный цикл эпизода ровно 3 действия (по 1 Step каждое):
  1. CODE  — полный python-скрипт (dual-branch: train+val + PREDICT=1 → predict).
  2. RUN   — запуск кода, env читает VAL_SCORE, регистрирует Candidate (cand_1).
  3. SUBMIT[CHOOSE:cand_1] — env применяет predict_code Кандидата к raw test.

Никакого Coach, никаких observations между шагами, никакой эскалации.
"""
from __future__ import annotations

from typing import Optional

from agent.llm_client import build_client
from core.types import Action, ActionType, Observation
from env.parser import parse_action

_SYSTEM = """\
Ты — ML-инженер. У тебя ОДНА попытка. Напиши ПОЛНЫЙ python-скрипт, который:
  1. Читает CSV из переменной окружения TRAIN_PATH (всегда есть колонка `target`).
  2. Делает train/val split, обучается, печатает строку ровно вида VAL_SCORE=<float>.
  3. Если ENV PREDICT=1 — читает TEST_PATH, применяет ТУ ЖЕ предобработку,
     пишет predictions.csv с одной колонкой (имя не важно).

Конвенции:
  * target колонка ВСЕГДА называется `target`.
  * категории через `pd.get_dummies(df, dummy_na=True)`, потом
    `test = test.reindex(columns=train_cols, fill_value=0)`.
  * NaN через `fillna(df.median(numeric_only=True))`.
  * sklearn ≥1.4: НЕ используй `sparse=` (только `sparse_output=False`),
    НЕ используй `squared=False` (только `np.sqrt(mean_squared_error(...))`).
  * Никаких lightgbm/xgboost (не та версия). Используй sklearn:
    RandomForestClassifier/Regressor или HistGradientBoostingClassifier/Regressor.

Ответь ровно одним блоком в формате:

[ACTION:code]
<full self-contained python script>
[/ACTION]
"""


def _build_prompt(obs: Observation) -> str:
    t = obs.task
    return (
        f"Задача: {t.description}\n"
        f"Метрика: {t.metric} ({'выше лучше' if t.metric_higher_better else 'ниже лучше'})\n"
        f"Бюджет токенов: {obs.tokens_left}\n\n"
        f"Напиши скрипт целиком, без диалога."
    )


class SingleShotAgent:
    """Один LLM-вызов → 3 предзаписанных Action [CODE, RUN, SUBMIT]."""

    def __init__(self, client=None, max_tokens: int = 4000) -> None:
        self.client = client if client is not None else build_client()
        self.max_tokens = max_tokens
        self._queue: list[Action] = []
        self.last_tokens: Optional[int] = None

    def act(self, obs: Observation) -> Action:
        if not self._queue:
            messages = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _build_prompt(obs)},
            ]
            raw = self.client.complete(messages, max_tokens=self.max_tokens)
            self.last_tokens = getattr(self.client, "last_usage_total", None)
            parsed = parse_action(raw)
            code = parsed.content if parsed.type == ActionType.CODE else raw
            self._queue = [
                Action(type=ActionType.CODE, content=code),
                Action(type=ActionType.RUN, content="single-shot run"),
                Action(type=ActionType.SUBMIT, content="[CHOOSE:cand_1]\nfinalize single-shot candidate"),
            ]
        return self._queue.pop(0)
