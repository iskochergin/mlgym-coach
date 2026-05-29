"""Демо-слайс end-to-end: Env + DummyCoach + BaselineAgent (MockLLM).

Замыкает петлю прогона на настоящем executor'е, но без реального LLM-ключа
(MockLLM) и без траты токенов — чтобы проверить стыковку
core.types ↔ env ↔ agent и что EpisodeResult пишется/читается.

Запуск из корня репозитория:
    python env/run_slice.py
или:
    python -m env.run_slice
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Гоним без реального ключа — детерминированные mock-ответы агента.
os.environ.setdefault("MLGYM_LLM", "mock")

from agent.baseline import BaselineAgent
from core.types import ActionType, EpisodeResult, Task
from env.executor import reset_executor
from env.gym import Env
from env.runner_adapter import resolve_coach

RUNS_DIR = _REPO_ROOT / "runs"


def _build_task() -> Task:
    return Task(
        id="breast_cancer_roc_auc",
        description="Бинарная классификация breast cancer. Метрика roc_auc, выше лучше.",
        metric="roc_auc",
        metric_higher_better=True,
        train_path=str(_REPO_ROOT / "tasks/data/breast_cancer_roc_auc/train.csv"),
        test_features_path=str(_REPO_ROOT / "tasks/data/breast_cancer_roc_auc/test_features.csv"),
    )


def run_episode(coach: str) -> EpisodeResult:
    reset_executor()
    task = _build_task()
    env = Env(
        task=task,
        coach=resolve_coach(coach),
        agent_name="baseline",
        seed=0,
    )
    agent = BaselineAgent()

    obs = env.reset()
    for _ in range(12):
        action = agent.act(obs)
        obs = env.step(action)
        if action.type == ActionType.SUBMIT:
            break
        if obs.tokens_left <= 0:
            break
    return env.result()


def _summary(coach: str, result: EpisodeResult) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"slice_demo_{coach}.json"
    out.write_text(result.to_json(), encoding="utf-8")

    back = EpisodeResult.from_json(out.read_text(encoding="utf-8"))
    total_hints = sum(len(s.hints) for s in back.steps)
    print(f"===== coach={coach!r} → wrote {out.relative_to(_REPO_ROOT)} =====")
    print(f"  steps            = {len(back.steps)}")
    print(f"  coverage         = {back.checklist_coverage}")
    print(f"  total hints      = {total_hints}")
    print(f"  final_test_score = {back.final_test_score}")
    print(f"  actions          = {[s.action.type.value for s in back.steps]}")
    print(f"  val_scores       = {[s.val_score for s in back.steps]}")
    print(f"  last_result      = {back.steps[-1].result!r}")
    print()


def main() -> None:
    # Прогон 1: дефолт (молчаливый коуч-заглушка) — hints не появляются.
    _summary("dummy", run_episode("dummy"))
    # Прогон 2: настоящий Coach — должны появиться hints, а грейдер посчитать test-метрику.
    _summary("real", run_episode("real"))


if __name__ == "__main__":
    main()
