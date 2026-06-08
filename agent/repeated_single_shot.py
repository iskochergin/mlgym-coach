"""RepeatedSingleShotAgent — N независимых попыток, между ними виден ТОЛЬКО
scalar val_score предыдущих (никаких observations, traceback, hints, кода).

Жизненный цикл (N=3 по умолчанию):
  ── round 1 ─── CODE → RUN  (env регистрирует cand_1 с val_score)
  ── round 2 ─── CODE → RUN  (cand_2; видит val_score cand_1)
  ── round 3 ─── CODE → RUN  (cand_3; видит val_scores предыдущих)
  ── final ──── SUBMIT      (env применяет best_by_validation; CHOOSE fallback)

Каждой попытке выдаётся budget = token_budget // N, чтобы общий расход не
превышал бюджет single_shot.
"""
from __future__ import annotations

from typing import Optional

from agent.llm_client import build_client
from core.types import Action, ActionType, Observation, Step
from env.parser import parse_action

_SYSTEM = """\
Ты — ML-инженер. У тебя НЕСКОЛЬКО независимых попыток. Каждый раз ты пишешь
ПОЛНЫЙ python-скрипт ЗАНОВО, как будто видишь задачу впервые. Между попытками
тебе дают ТОЛЬКО список val_score предыдущих — никакого кода, traceback,
подсказок. Цели:
  * не повторять явно плохие подходы (например, если val_score прошлой попытки
    был очень низкий, попробуй другую модель/preprocessing);
  * попыток мало — не уходи в маргинальные сложности, делай простое и рабочее.

Скрипт должен:
  1. Прочитать TRAIN_PATH (всегда есть колонка `target`).
  2. Сделать train/val split, обучиться, напечатать VAL_SCORE=<float>.
  3. Если PREDICT=1 — прочитать TEST_PATH, применить ту же предобработку,
     записать predictions.csv (одна колонка).

Конвенции:
  * target ВСЕГДА `target`.
  * `pd.get_dummies(df, dummy_na=True)` + `test.reindex(columns=train_cols, fill_value=0)`.
  * sklearn ≥1.4: `sparse_output=False`, RMSE через `np.sqrt(mean_squared_error(...))`.
  * Только pandas/numpy/sklearn. Никаких lightgbm/xgboost.

Ответь ровно одним блоком:

[ACTION:code]
<full python script>
[/ACTION]
"""


class RepeatedSingleShotAgent:
    """N независимых попыток. На каждой — один LLM-вызов и [CODE, RUN].
    После N раундов — SUBMIT (env возьмёт best Candidate)."""

    def __init__(self, client=None, n_attempts: int = 3, token_budget: int = 30000) -> None:
        self.client = client if client is not None else build_client()
        self.n_attempts = int(n_attempts)
        self.budget_per_attempt = max(500, token_budget // self.n_attempts)
        self._queue: list[Action] = []
        self._attempts_done: int = 0
        self.last_tokens: Optional[int] = None

    def _build_prompt(self, obs: Observation, prev_scores: list[Optional[float]]) -> str:
        t = obs.task
        prev = ", ".join(f"{s:.4f}" if s is not None else "FAIL" for s in prev_scores) or "(пока нет)"
        return (
            f"Задача: {t.description}\n"
            f"Метрика: {t.metric} ({'выше лучше' if t.metric_higher_better else 'ниже лучше'})\n"
            f"Попытка {self._attempts_done + 1} из {self.n_attempts}.\n"
            f"VAL_SCORE предыдущих попыток: [{prev}]\n\n"
            f"Напиши скрипт целиком, без диалога."
        )

    def _collect_prev_scores(self, history: list[Step]) -> list[Optional[float]]:
        # Берём val_score из RUN-шагов в порядке появления.
        return [s.val_score for s in history if s.action.type == ActionType.RUN]

    def act(self, obs: Observation) -> Action:
        if self._queue:
            return self._queue.pop(0)

        if self._attempts_done >= self.n_attempts:
            # Финал — env применит best_by_validation.
            return Action(
                type=ActionType.SUBMIT,
                content="[CHOOSE:best]\nrepeated single-shot finalize",
            )

        prev_scores = self._collect_prev_scores(obs.history)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": self._build_prompt(obs, prev_scores)},
        ]
        raw = self.client.complete(messages, max_tokens=self.budget_per_attempt)
        self.last_tokens = getattr(self.client, "last_usage_total", None)
        parsed = parse_action(raw)
        code = parsed.content if parsed.type == ActionType.CODE else raw

        self._attempts_done += 1
        self._queue = [
            Action(type=ActionType.CODE, content=code),
            Action(type=ActionType.RUN, content=f"attempt {self._attempts_done}"),
        ]
        return self._queue.pop(0)
