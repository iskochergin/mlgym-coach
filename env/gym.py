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
from env.candidates import CandidateRegistry, parse_choose_marker
from env.dummy_coach import DummyCoach
from env.executor import run_solution
from env.stage_policy import StagePolicy, resolve_stage_policy

ENV_VERSION = "v3-pr1"

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
        max_steps: Optional[int] = None,
        stage_policy: Optional[StagePolicy] = None,
        mode: str = "flexible",
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
        # Нужно, чтобы знать, сколько шагов осталось — для принудительного submit.
        self.max_steps: Optional[int] = max_steps
        # Mode и stage_policy — для 4-режимного challenge.
        self.mode: str = mode
        self.stage_policy: StagePolicy = stage_policy or resolve_stage_policy("flexible")
        self._policy_applies = mode in {"fixed", "flexible"}

        self._history: list[Step] = []
        self._stage: Stage = Stage.EDA  # стартовая стадия (UNDERSTAND убрали в Stage 5→4)
        self._tokens_left: int = token_budget
        self._last_val_score: Optional[float] = None
        self._last_result: Optional[str] = None
        self._final_test_score: Optional[float] = None
        self._last_coverage: float = 0.0
        self._has_run: bool = False
        self._current_code: str = ""  # последний принятый CODE — его исполняют RUN/SUBMIT
        self._step_count: int = 0
        self._forced_submit_done: bool = False
        # PR1: candidate registry + grade lock + failure modes.
        self._candidates = CandidateRegistry(metric_higher_better=task.metric_higher_better)
        self._grade_count: int = 0   # сколько раз дёрнули grader
        self._locked: bool = False   # после первой оценки — лок дальнейших submit
        self._failure_modes: list[str] = []
        self._chosen_candidate_id: Optional[str] = None

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
        self._step_count = 0
        self._forced_submit_done = False
        self._candidates = CandidateRegistry(metric_higher_better=self.task.metric_higher_better)
        self._grade_count = 0
        self._locked = False
        self._failure_modes = []
        self._chosen_candidate_id = None
        return Observation(
            task=self.task,
            stage=self._stage,
            history=[],
            last_result=None,
            val_score=None,
            tokens_left=self._tokens_left,
            hints=[],
        )

    def step(self, action: Action, tokens_used: Optional[int] = None) -> Observation:
        # Сохраняем ОРИГИНАЛЬНЫЙ content до подмены, чтобы оценка по эвристике
        # (когда real tokens_used отсутствует) считалась по реальному действию агента.
        original_content_len = len(action.content)

        # Stage policy: применяется ТОЛЬКО в fixed/flexible режимах. Для single_shot
        # и repeated_single_shot стадии нерелевантны (one-shot выполнение).
        if self._policy_applies:
            allowed, replacement = self.stage_policy.check(self._stage, action)
            if not allowed and replacement is not None:
                action = replacement

        # Plan-spam guard: если последние 2 действия были PLAN и текущее тоже PLAN —
        # модель залипла. Подменяем по приоритету: есть val_score → SUBMIT;
        # есть код но без скор → RUN; иначе пропускаем (даём дойти до nudge'а).
        last_two = [s.action.type for s in self._history[-2:]]
        is_plan_loop = (
            action.type == ActionType.PLAN
            and len(last_two) == 2
            and all(t == ActionType.PLAN for t in last_two)
        )
        if is_plan_loop and self._last_val_score is not None and self._current_code:
            action = Action(
                type=ActionType.SUBMIT,
                content="[FORCED_SUBMIT: plan-loop ≥3 + val_score есть → финализирую]",
            )
            self._forced_submit_done = True
        elif is_plan_loop and self._current_code:
            action = Action(
                type=ActionType.RUN,
                content="[FORCED_RUN: plan-loop ≥3 → запускаю текущий код]",
            )

        # Force-submit: если уже есть val_score и осталось ≤2 шагов до max_steps,
        # а агент опять выдаёт не-submit/не-run — финализируем эпизод СУЩЕСТВУЮЩИМ
        # кодом, чтобы не уходить в None final_test_score.
        if (
            self.max_steps is not None
            and self.max_steps >= 4
            and self._last_val_score is not None
            and self._current_code
            and not self._forced_submit_done
            and self._step_count >= self.max_steps - 2
            and action.type not in (ActionType.SUBMIT, ActionType.RUN)
        ):
            action = Action(
                type=ActionType.SUBMIT,
                content="[FORCED_SUBMIT: steps_left<=2 + val_score есть → финализирую текущий код]",
            )
            self._forced_submit_done = True

        if tokens_used is None:
            tokens_used = max(1, original_content_len // 4)
        self._tokens_left -= tokens_used
        self._step_count += 1

        new_stage = self._next_stage(self._stage, action.type)
        result, val_score = self._execute(action)

        # _has_run выставляем ПОСЛЕ _next_stage, чтобы текущий RUN
        # не сдвигал стадию (двигаются только следующие CODE-шаги).
        if action.type == ActionType.RUN:
            self._has_run = True
            self._last_val_score = val_score
            # PR1: автоматическая регистрация Candidate на успешном RUN.
            # predict_code = _current_code (требуется dual-branch: train + PREDICT=1).
            if val_score is not None and self._current_code:
                self._candidates.register(
                    predict_code=self._current_code,
                    val_score=val_score,
                    step_idx=len(self._history),
                )

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
        # Если коуч обёрнут (например HintLevelLimitedCoach) — пишем имя ВНУТРЕННЕГО
        # коуча, чтобы дашборд группировал прогоны по содержательному типу, а не
        # обёртке. Плюс отдельным полем фиксируем max_hint_level для ablation-разбора.
        inner_coach = getattr(self.coach, "inner", self.coach)
        coach_cfg: dict = {"coach": type(inner_coach).__name__}
        if hasattr(self.coach, "max_level"):
            coach_cfg["max_hint_level"] = int(getattr(self.coach, "max_level"))
        # PR1: snapshot env config (env_version, mode, stage_policy, sandbox),
        # failure_modes, candidates registry, chosen candidate.
        snapshot_cfg: dict = {
            "token_budget": self.token_budget,
            "max_steps": self.max_steps,
            "env_version": ENV_VERSION,
            "mode": self.mode,
            "stage_policy": getattr(self.stage_policy, "name", "flexible"),
            "sandbox": os.environ.get("MLGYM_SANDBOX", "permissive"),
            **coach_cfg,
            "failure_modes": list(self._failure_modes),
            "candidates_n": len(self._candidates.all()),
            "chosen_candidate_id": self._chosen_candidate_id,
            "grade_count": self._grade_count,
        }
        return EpisodeResult(
            task_id=self.task.id,
            agent=self.agent_name,
            seed=self.seed,
            steps=list(self._history),
            final_test_score=self._final_test_score,
            checklist_coverage=self._last_coverage,
            total_tokens=total_tokens,
            config=snapshot_cfg,
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
            return self._execute_submit_with_candidate(action)
        if action.type == ActionType.CODE:
            self._current_code = action.content
            lines = action.content.count("\n") + 1
            return f"code accepted ({lines} lines)", None
        if action.type == ActionType.PLAN:
            return "plan noted", None
        if action.type == ActionType.EDA:
            return "eda noted", None
        return "", None

    def _execute_submit_with_candidate(self, action: Action) -> tuple[str, Optional[float]]:
        """SUBMIT: применяет predict_code Кандидата (явно выбранного через
        [CHOOSE:cand_id] или fallback на best_by_validation) к raw test rows.

        Privacy lock: после первой оценки эпизод залочен. Повторный SUBMIT
        возвращает None и пишет failure_mode 'double_final_grade_attempt'."""
        # Privacy lock — повторная оценка запрещена.
        if self._locked:
            self._failure_modes.append("double_final_grade_attempt")
            return "[LOCKED] эпизод уже оценён; повторный SUBMIT отклонён", None

        # Разбираем [CHOOSE:cand_id] (опциональный префикс в content).
        cid = parse_choose_marker(action.content)
        chosen = None
        if cid:
            if cid.lower() == "best":
                chosen = self._candidates.best_by_validation()
            else:
                chosen = self._candidates.get(cid)
            if chosen is None and cid.lower() != "best":
                # Невалидный candidate_id — отметим, fallback на best.
                self._failure_modes.append(f"invalid_choose_id:{cid}")
                chosen = self._candidates.best_by_validation()
            if chosen is not None:
                self._chosen_candidate_id = chosen.candidate_id

        # Fallback: нет явного CHOOSE → best_by_validation. Если и тот None —
        # все попытки провалились, эпизод финализируется без скора.
        if chosen is None:
            chosen = self._candidates.best_by_validation()
            if chosen is not None:
                self._chosen_candidate_id = chosen.candidate_id

        if chosen is None:
            self._failure_modes.append("no_candidate_registered")
            self._locked = True
            self._final_test_score = None
            return "[NO_CANDIDATE] не зарегистрировано ни одного валидного кандидата", None

        result, score = self._execute_submit(chosen.predict_code)
        # Lock после первой попытки оценки (успешной или нет).
        self._grade_count += 1
        self._locked = True
        return result, score

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
