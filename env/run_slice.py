"""Демо-слайс end-to-end: Env + DummyCoach + ScriptedAgent.

Замыкает петлю прогона на заглушках, чтобы убедиться, что интерфейсы
core.types ↔ env ↔ agent состыкованы и EpisodeResult пишется/читается.

Запуск из корня репозитория:
    python env/run_slice.py
или:
    python -m env.run_slice
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.scripted import ScriptedAgent
from core.types import Action, ActionType, EpisodeResult, Task
from env.dummy_coach import DummyCoach
from env.executor import reset_executor
from env.gym import Env

RUNS_DIR = _REPO_ROOT / "runs"
OUTPUT_PATH = RUNS_DIR / "slice_demo.json"


def _build_task() -> Task:
    return Task(
        id="churn_small",
        description="Бинарная классификация churn_small. Метрика roc_auc.",
        metric="roc_auc",
        metric_higher_better=True,
        train_path="data/churn_small/train.csv",
        test_features_path="data/churn_small/test_features.csv",
    )


def _scripted_actions() -> list[Action]:
    return [
        Action(
            type=ActionType.PLAN,
            content="binary classification churn_small; baseline a tree, then improve, then submit",
        ),
        Action(
            type=ActionType.CODE,
            content="model = GradientBoostingClassifier(random_state=0).fit(X, y)",
        ),
        Action(type=ActionType.RUN, content="run current code"),
        Action(
            type=ActionType.CODE,
            content="model = LGBMClassifier(n_estimators=500, learning_rate=0.05).fit(X, y)",
        ),
        Action(type=ActionType.RUN, content="run current code"),
        Action(
            type=ActionType.SUBMIT,
            content="proba = model.predict_proba(X_test)[:, 1]; submit(proba)",
        ),
    ]


def main() -> None:
    reset_executor()
    task = _build_task()
    env = Env(task=task, coach=DummyCoach(), agent_name="baseline", seed=0)
    agent = ScriptedAgent(actions=_scripted_actions())

    obs = env.reset()
    while True:
        action = agent.act(obs)
        obs = env.step(action)
        if action.type == ActionType.SUBMIT:
            break
        if obs.tokens_left <= 0:
            break

    result = env.result()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(result.to_json(), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(_REPO_ROOT)}")

    # Round-trip + сводка.
    back = EpisodeResult.from_json(OUTPUT_PATH.read_text(encoding="utf-8"))
    print()
    print("summary (re-parsed via EpisodeResult.from_json):")
    print(f"  agent           = {back.agent!r}")
    print(f"  steps           = {len(back.steps)}")
    print(f"  final_test_score = {back.final_test_score}")
    print(f"  total_tokens    = {back.total_tokens}")
    print(f"  stages          = {[s.stage.value for s in back.steps]}")
    print(f"  val_scores      = {[s.val_score for s in back.steps]}")


if __name__ == "__main__":
    main()
