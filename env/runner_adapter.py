"""Прослойка между раннером Софы и средой env.gym.Env.

runner/env_factory.build_env ожидает объект с методом .run() -> EpisodeResult
(см. протокол RunnerEnv). Наш Env работает в стиле reset/step/result, поэтому
здесь оборачиваем «Env + агент + петля» в один .run().

Зоны не нарушаем: runner/ не трогаем. Чтобы переключить раннер на настоящую
среду, Софе достаточно в runner/env_factory.py для ветки env_name == "real"
вернуть build_real_env(...) отсюда (одна строка — см. заметку в конце задачи).
"""
from __future__ import annotations

import sys
from typing import Optional

from agent.baseline import BaselineAgent
from core.types import Action, ActionType, EpisodeResult, Task
from env.executor import reset_executor
from env.gym import Env


class HintLevelLimitedCoach:
    """Обёртка над любым коучем, отсекающая подсказки выше указанного уровня.

    Нужна для ablation-экспериментов ('какой уровень подсказок реально помогает').
    Сам внутренний коуч продолжает считать coverage по полному чек-листу и
    эскалировать level — мы только фильтруем то, что доходит до агента."""

    def __init__(self, inner, max_level: int = 3) -> None:
        self.inner = inner
        # Зажимаем в [0, 3] — за этими границами обёртка теряет смысл.
        # 0 = «без подсказок», 1/2/3 = ablation-уровни.
        self.max_level = max(0, min(3, int(max_level)))

    def assess(self, obs):
        coverage, hints = self.inner.assess(obs)
        filtered = []
        for h in hints:
            try:
                lvl = int(getattr(h, "level", 0))
            except (TypeError, ValueError):
                continue  # кастомный коуч с битым level — отсекаем
            if lvl <= self.max_level:
                filtered.append(h)
        return coverage, filtered


def resolve_coach(coach, *, max_hint_level: int = 3):
    """Привести спецификацию коуча к объекту с .assess(obs) -> (coverage, hints).

    Принимает:
      - None / "dummy"  → DummyCoach() (молчаливый коуч-заглушка);
      - "real"          → coach.coach.Coach() (ленивый импорт; env не тащит
                          зависимость от coach/ на верхнем уровне);
      - готовый объект  → возвращается как есть.
    Если настоящий коуч недоступен (coach/ ещё нет, ошибка импорта и т.п.) —
    fallback на DummyCoach с предупреждением в stderr, без падения.

    max_hint_level (1..3): фильтрует подсказки настоящего коуча. Для DummyCoach
    бесполезен (он и так молчит), но логически тоже применяется."""
    if coach is None or coach == "dummy":
        from env.dummy_coach import DummyCoach

        inner = DummyCoach()
    elif coach == "real":
        try:
            from coach.coach import Coach

            inner = Coach()
        except Exception as e:
            print(
                f"[runner_adapter] настоящий Coach недоступен ({e!r}); fallback на DummyCoach",
                file=sys.stderr,
            )
            from env.dummy_coach import DummyCoach

            inner = DummyCoach()
    else:
        inner = coach  # уже готовый объект-коуч
    return inner if int(max_hint_level) >= 3 else HintLevelLimitedCoach(inner, max_hint_level)


class RealEnv:
    """RunnerEnv-совместимая обёртка: гоняет ReAct-петлю до SUBMIT/лимитов."""

    def __init__(
        self,
        *,
        task: Task,
        agent: str,
        seed: int,
        config,
        coach="dummy",
        agent_obj=None,
        token_budget: Optional[int] = None,
        hidden_labels_path: Optional[str] = None,
        run_dir: Optional[str] = None,
        max_hint_level: int = 3,
    ) -> None:
        self.task = task
        self.agent = agent
        self.seed = seed
        self.config = config
        # coach: "dummy" | "real" | готовый объект-коуч.
        self._coach = resolve_coach(coach, max_hint_level=max_hint_level)
        self._agent = agent_obj if agent_obj is not None else BaselineAgent()
        self._max_steps = int(getattr(config, "max_steps", 12))
        # Явный token_budget важнее значения из config (дефолт — из config / 50k).
        self._token_budget = int(
            token_budget if token_budget is not None else getattr(config, "token_budget", 50_000)
        )
        self._hidden_labels_path = hidden_labels_path
        self._run_dir = run_dir

    def run(self) -> EpisodeResult:
        reset_executor()
        env = Env(
            task=self.task,
            coach=self._coach,
            token_budget=self._token_budget,
            seed=self.seed,
            agent_name=self.agent,
            hidden_labels_path=self._hidden_labels_path,
            run_dir=self._run_dir,
            max_steps=self._max_steps,
        )
        obs = env.reset()
        for i in range(self._max_steps):
            action = self._agent.act(obs)
            real_tokens = getattr(self._agent, "last_tokens", None)
            obs = env.step(action, tokens_used=real_tokens)
            executed = obs.history[-1].action if obs.history else action
            if executed.type == ActionType.SUBMIT:
                break
            if obs.tokens_left <= 0:
                break
            # Last-chance submit: получили val_score на одном из финальных шагов —
            # не отдаём решение модели, форсим submit сами, чтобы эпизод не закончился None.
            last_val = obs.history[-1].val_score if obs.history else None
            last_run_ok = (
                executed.type == ActionType.RUN
                and last_val is not None
                and last_val == last_val  # NaN-проверка: float('nan') != float('nan')
            )
            steps_left = self._max_steps - (i + 1)
            if last_run_ok and steps_left <= 1:
                forced = Action(
                    type=ActionType.SUBMIT,
                    content="[LOOP_FORCED_SUBMIT: val_score есть, шагов в обрез]",
                )
                obs = env.step(forced)
                break
        return env.result()


def build_real_env(
    *,
    task: Task,
    agent: str,
    seed: int,
    config,
    coach: str = "dummy",
    token_budget: Optional[int] = None,
    hidden_labels_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    max_hint_level: int = 3,
) -> RealEnv:
    """Фабрика для runner/env_factory (ветка env_name == 'real').

    coach: "dummy" (по умолчанию) | "real" | готовый объект-коуч.
    token_budget: явный бюджет (важнее значения из config).
    hidden_labels_path: явный путь к y_test (важнее соглашения по task.id).
    run_dir: куда писать снапшоты эпизода (partial + финальный)."""
    return RealEnv(
        task=task,
        agent=agent,
        seed=seed,
        config=config,
        coach=coach,
        token_budget=token_budget,
        hidden_labels_path=hidden_labels_path,
        run_dir=run_dir,
        max_hint_level=max_hint_level,
    )
