"""Gym-петля: Env(task, coach) с reset/step/result.

Интерфейс минимальный, чтобы:
  - агент видел только Observation,
  - результат укладывался в core.types.EpisodeResult,
  - coach можно было подменить любым объектом с .assess(obs) -> (coverage, hints).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
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

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_hidden_labels(task: Task) -> Optional[str]:
    """Путь к скрытым лейблам по соглашению Софы (tasks/hidden_labels/<id>/y_test.csv).

    Task (заморожен) поля hidden_labels_path не содержит, поэтому выводим путь
    из task.id. Если файла нет — None (грейдер просто пропускается)."""
    candidate = _REPO_ROOT / "tasks" / "hidden_labels" / task.id / "y_test.csv"
    return str(candidate) if candidate.exists() else None


class _CoachLike(Protocol):
    def assess(self, obs: Observation) -> tuple[float, list[Hint]]: ...


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
        hidden_labels_path: Optional[str] = None,
        run_dir: Optional[str] = None,
    ) -> None:
        self.task = task
        self.coach: _CoachLike = coach if coach is not None else DummyCoach()
        self.token_budget = token_budget
        self.seed = seed
        self.agent_name = agent_name
        # Приоритет hidden_labels: явный параметр > соглашение по task.id > None.
        self.hidden_labels_path: Optional[str] = (
            hidden_labels_path if hidden_labels_path is not None else _default_hidden_labels(task)
        )
        # Куда писать снапшоты эпизода (partial во время прогона + финальный).
        self.run_dir: Path = Path(run_dir) if run_dir is not None else (_REPO_ROOT / "runs" / "local")

        self._history: list[Step] = []
        self._stage: Stage = Stage.EDA  # стартовая стадия (UNDERSTAND убрали в Stage 5→4)
        self._tokens_left: int = token_budget
        self._last_val_score: Optional[float] = None
        self._last_result: Optional[str] = None
        self._final_test_score: Optional[float] = None
        self._last_coverage: float = 0.0
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
        coverage, hints = self.coach.assess(obs)
        self._last_coverage = coverage

        step.hints = list(hints)
        obs.hints = list(hints)

        self._history.append(step)
        # Инкрементальный снапшот: дашборд, тейлящий файл, видит прогресс вживую.
        self._write_partial()
        return obs

    def _episode_snapshot(self) -> EpisodeResult:
        """Текущий снапшот эпизода (история + накопленные токены/покрытие)."""
        total_tokens = sum(s.tokens_used for s in self._history)
        return EpisodeResult(
            task_id=self.task.id,
            agent=self.agent_name,
            seed=self.seed,
            steps=list(self._history),
            final_test_score=self._final_test_score,
            checklist_coverage=self._last_coverage,
            total_tokens=total_tokens,
            config={
                "token_budget": self.token_budget,
                "coach": type(self.coach).__name__,
            },
        )

    def result(self) -> EpisodeResult:
        """Финализировать эпизод: записать episode.json, удалить partial."""
        snapshot = self._episode_snapshot()
        self._write_final(snapshot)
        return snapshot

    # ─────────────────────────── snapshot I/O ───────────────────────────

    def _partial_path(self) -> Path:
        return self.run_dir / "episode.partial.json"

    def _final_path(self) -> Path:
        return self.run_dir / "episode.json"

    def _atomic_write(self, path: Path, text: str) -> None:
        """Запись через временный файл + os.replace (атомарная подмена),
        чтобы читатель не поймал полузаписанный JSON."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _write_partial(self) -> None:
        try:
            self._atomic_write(self._partial_path(), self._episode_snapshot().to_json())
        except Exception as e:  # запись снапшота не должна валить эпизод
            print(f"[env.gym] не удалось записать partial-снапшот: {e!r}", file=sys.stderr)

    def _write_final(self, snapshot: EpisodeResult) -> None:
        try:
            self._atomic_write(self._final_path(), snapshot.to_json())
            partial = self._partial_path()
            if partial.exists():
                partial.unlink()
        except Exception as e:
            print(f"[env.gym] не удалось записать финальный снапшот: {e!r}", file=sys.stderr)

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
            return self._execute_submit(self._current_code or action.content)
        if action.type == ActionType.CODE:
            self._current_code = action.content
            lines = action.content.count("\n") + 1
            return f"code accepted ({lines} lines)", None
        if action.type == ActionType.PLAN:
            return "plan noted", None
        if action.type == ActionType.EDA:
            return "eda noted", None
        return "", None

    def _execute_submit(self, code: str) -> tuple[str, Optional[float]]:
        """Песочница пишет predictions.csv → грейдер считает test-метрику по y_test.

        Грейдер живёт в coach/ — импортируем лениво, чтобы env не тащил эту
        зависимость на верхнем уровне. Любой сбой грейдера логируем в stderr и
        оставляем final_test_score = None, эпизод не валим.
        Возвращаем test_score как score шага (виден в Step.val_score) и кладём
        его же в self._final_test_score (→ EpisodeResult.final_test_score)."""
        fd, preds_path = tempfile.mkstemp(prefix="mlgym_pred_", suffix=".csv")
        os.close(fd)
        try:
            result, _ = run_solution(
                code, self.task, mode="submit", predictions_out=preds_path
            )

            if not os.path.exists(preds_path) or os.path.getsize(preds_path) == 0:
                # Песочница не записала предсказания (ошибка/таймаут) — без грейдинга.
                self._final_test_score = None
                return result, None

            if not self.hidden_labels_path:
                print(
                    f"[env.gym] grader пропущен: нет hidden_labels для task {self.task.id!r}",
                    file=sys.stderr,
                )
                self._final_test_score = None
                return result, None

            try:
                from coach.grader import score_submission

                test_score = float(
                    score_submission(self.task, preds_path, self.hidden_labels_path)
                )
            except Exception as e:  # грейдер не должен ронять эпизод
                print(f"[env.gym] grader упал: {e!r}", file=sys.stderr)
                self._final_test_score = None
                return result, None

            self._final_test_score = test_score
            return f"{result}; test {self.task.metric}={test_score:.4f}", test_score
        finally:
            try:
                os.remove(preds_path)
            except OSError:
                pass
