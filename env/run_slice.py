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


_TASKS = {
    "breast_cancer_roc_auc": Task(
        id="breast_cancer_roc_auc",
        description="Бинарная классификация breast cancer. Метрика roc_auc, выше лучше.",
        metric="roc_auc",
        metric_higher_better=True,
        train_path=str(_REPO_ROOT / "tasks/data/breast_cancer_roc_auc/train.csv"),
        test_features_path=str(_REPO_ROOT / "tasks/data/breast_cancer_roc_auc/test_features.csv"),
    ),
    "diabetes_rmse": Task(
        id="diabetes_rmse",
        description="Регрессия по диабетическому датасету sklearn. Метрика RMSE, ниже лучше.",
        metric="rmse",
        metric_higher_better=False,
        train_path=str(_REPO_ROOT / "tasks/data/diabetes_rmse/train.csv"),
        test_features_path=str(_REPO_ROOT / "tasks/data/diabetes_rmse/test_features.csv"),
    ),
}


def _default_max_steps(metric: str) -> int:
    # Регрессия даёт больше пространства для итераций → выше дефолт.
    return 15 if metric.lower() in {"rmse", "mae"} else 10


def run_episode(coach: str, task_id: str = "breast_cancer_roc_auc", *, token_budget: int = 50_000) -> EpisodeResult:
    reset_executor()
    task = _TASKS[task_id]
    run_dir = RUNS_DIR / f"slice_{coach}_{task_id}"
    hidden_labels = _REPO_ROOT / f"tasks/hidden_labels/{task_id}/y_test.csv"
    max_steps = _default_max_steps(task.metric)
    env = Env(
        task=task,
        coach=resolve_coach(coach),
        agent_name="baseline",
        seed=0,
        token_budget=token_budget,
        hidden_labels_path=str(hidden_labels),
        run_dir=str(run_dir),
        max_steps=max_steps,
    )
    agent = BaselineAgent()

    obs = env.reset()
    for _ in range(max_steps):
        action = agent.act(obs)
        obs = env.step(action, tokens_used=getattr(agent, "last_tokens", None))
        executed = obs.history[-1].action if obs.history else action
        if executed.type == ActionType.SUBMIT:
            break
        if obs.tokens_left <= 0:
            break
    return env.result()


def _summary(coach: str, task_id: str, result: EpisodeResult) -> None:
    run_dir = RUNS_DIR / f"slice_{coach}_{task_id}"
    total_hints = sum(len(s.hints) for s in result.steps)
    partial = run_dir / "episode.partial.json"
    final = run_dir / "episode.json"
    print(f"===== coach={coach!r} task={task_id} → {run_dir.relative_to(_REPO_ROOT)}/ =====")
    print(f"  episode.json exists      = {final.exists()}")
    print(f"  episode.partial removed  = {not partial.exists()}")
    print(f"  steps            = {len(result.steps)}")
    print(f"  coverage         = {result.checklist_coverage}")
    print(f"  total hints      = {total_hints}")
    print(f"  total_tokens     = {result.total_tokens}")
    print(f"  final_test_score = {result.final_test_score}")
    print(f"  actions          = {[s.action.type.value for s in result.steps]}")
    print(f"  val_scores       = {[s.val_score for s in result.steps]}")
    print(f"  last_result      = {result.steps[-1].result!r}")
    print()


def main() -> None:
    # 1) breast_cancer baseline — dummy & real coach (как раньше).
    _summary("dummy", "breast_cancer_roc_auc", run_episode("dummy", "breast_cancer_roc_auc"))
    _summary("real", "breast_cancer_roc_auc", run_episode("real", "breast_cancer_roc_auc"))
    # 2) diabetes — проверка force-submit: regression раньше не доходил до SUBMIT.
    _summary("dummy", "diabetes_rmse", run_episode("dummy", "diabetes_rmse"))


if __name__ == "__main__":
    main()
