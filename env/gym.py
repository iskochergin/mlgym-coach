"""Gym-петля: Env(task, coach) с reset/step/result.

Интерфейс минимальный, чтобы:
  - агент видел только Observation,
  - результат укладывался в core.types.EpisodeResult,
  - coach можно было подменить любым объектом с .assess(obs) -> (coverage, hints).
"""
from __future__ import annotations

from typing import Optional, Protocol

from core.types import (
    Action,
    ActionType,
    EpisodeResult,
    Hint,
    Observation,
    Stage,
    Step,
    Task,
)
from env.dummy_coach import DummyCoach
from env.executor import run_solution


class _CoachLike(Protocol):
    def assess(self, obs: Observation) -> tuple[float, dict[str, float], list[Hint]]: ...


class Env:
    """v1 среда: стадии переключаются по типу действия, бюджет токенов конечен,
    исполнение кода — через executor.run_solution (пока заглушка).
    """

    def __init__(
        self,
        task: Task,
        coach: Optional[_CoachLike] = None,
        *,
        token_budget: int = 50_000,
        seed: int = 0,
        agent_name: str = "baseline",
    ) -> None:
        self.task = task
        self.coach: _CoachLike = coach if coach is not None else DummyCoach()
        self.token_budget = token_budget
        self.seed = seed
        self.agent_name = agent_name

        self._history: list[Step] = []
        self._stage: Stage = Stage.EDA  # стартовая стадия (UNDERSTAND убрали в Stage 5→4)
        self._tokens_left: int = token_budget
        self._last_val_score: Optional[float] = None
        self._last_result: Optional[str] = None
        self._final_test_score: Optional[float] = None
        self._last_coverage: float = 0.0
        self._stage_coverage: dict[str, float] = {}
        self._has_run: bool = False
        self._current_code: str = ""  # последний принятый CODE — его исполняют RUN/SUBMIT

    def reset(self, task: Optional[Task] = None) -> Observation:
        if task is not None:
            self.task = task
        self._history = []
        self._stage = Stage.EDA
        self._tokens_left = self.token_budget
        self._last_val_score = None
        self._last_result = None
        self._final_test_score = None
        self._last_coverage = 0.0
        self._stage_coverage = {}
        self._has_run = False
        self._current_code = ""
        return Observation(
            task=self.task,
            stage=self._stage,
            history=[],
            last_result=None,
            val_score=None,
            tokens_left=self._tokens_left,
            hints=[],
        )

    def step(self, action: Action) -> Observation:
        tokens_used = max(1, len(action.content) // 4)
        self._tokens_left -= tokens_used

        new_stage = self._next_stage(self._stage, action.type)
        result, val_score = self._execute(action)

        # _has_run выставляем ПОСЛЕ _next_stage, чтобы текущий RUN
        # не сдвигал стадию (двигаются только следующие CODE-шаги).
        if action.type == ActionType.RUN:
            self._has_run = True
            self._last_val_score = val_score

        self._stage = new_stage
        self._last_result = result

        step = Step(
            idx=len(self._history),
            stage=new_stage,
            action=action,
            result=result,
            val_score=val_score,
            tokens_used=tokens_used,
            hints=[],
        )

        # Коучу даём obs, в истории которой уже виден только что выполненный шаг.
        obs = Observation(
            task=self.task,
            stage=new_stage,
            history=self._history + [step],
            last_result=result,
            val_score=self._last_val_score,
            tokens_left=self._tokens_left,
            hints=[],
        )
        coverage, stage_coverage, hints = self.coach.assess(obs)
        self._last_coverage = coverage

        step.hints = list(hints)
        obs.hints = list(hints)

        self._last_coverage = coverage
        self._stage_coverage = stage_coverage

        self._history.append(step)
        return obs

    def result(self) -> EpisodeResult:
        total_tokens = sum(s.tokens_used for s in self._history)
        return EpisodeResult(
            task_id=self.task.id,
            agent=self.agent_name,
            seed=self.seed,
            steps=list(self._history),
            final_test_score=self._final_test_score,
            checklist_coverage=self._last_coverage,
            stage_coverage=self._stage_coverage,
            total_tokens=total_tokens,
            config={
                "token_budget": self.token_budget,
                "coach": type(self.coach).__name__,
            },
        )

    # ─────────────────────────── helpers ───────────────────────────

    def _next_stage(self, current: Stage, action_type: ActionType) -> Stage:
        if action_type == ActionType.PLAN:
            return current  # PLAN не двигает стадию (раньше вёл в UNDERSTAND)
        if action_type == ActionType.EDA:
            return Stage.EDA
        if action_type == ActionType.CODE:
            return Stage.IMPROVE if self._has_run else Stage.BASELINE
        if action_type == ActionType.RUN:
            return current
        if action_type == ActionType.SUBMIT:
            return Stage.SUBMIT
        return current

    def _execute(self, action: Action) -> tuple[str, Optional[float]]:
        """Вернуть (result, val_score) для шага. val_score только у RUN.

        RUN/SUBMIT исполняют последний принятый CODE (self._current_code),
        а не собственный content действия (там обычно лишь «run current code»).
        """
        if action.type == ActionType.RUN:
            if not self._current_code:
                return "нет кода для запуска: сначала пришли действие CODE", None
            return run_solution(self._current_code, self.task, mode="run")
        if action.type == ActionType.SUBMIT:
            code = self._current_code or action.content
            result, test_score = run_solution(code, self.task, mode="submit")
            self._final_test_score = test_score
            # Step.val_score держим только для валидации — у SUBMIT-шага оставляем None,
            # а test-скор живёт в EpisodeResult.final_test_score.
            return result, None
        if action.type == ActionType.CODE:
            self._current_code = action.content
            lines = action.content.count("\n") + 1
            return f"code accepted ({lines} lines)", None
        if action.type == ActionType.PLAN:
            return "plan noted", None
        if action.type == ActionType.EDA:
            return "eda noted", None
        return "", None
