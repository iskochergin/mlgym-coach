"""Генерация двух примеров EpisodeResult (baseline и scaffold) на одной и той же
фейковой задаче churn_small. Источник правды по формату — core.types.

Запуск из корня репозитория:
    python -m examples.make_examples
или:
    python examples/make_examples.py

Кладёт результаты в examples/episode_baseline.json и examples/episode_scaffold.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы скрипт работал и через `python examples/make_examples.py`, и через `python -m examples.make_examples`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.types import (
    Action,
    ActionType,
    EpisodeResult,
    Hint,
    Stage,
    Step,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
BASELINE_PATH = EXAMPLES_DIR / "episode_baseline.json"
SCAFFOLD_PATH = EXAMPLES_DIR / "episode_scaffold.json"
SCAFFOLD_SEED1_PATH = EXAMPLES_DIR / "episode_scaffold_seed1.json"

TASK_ID = "churn_small"


def build_baseline() -> EpisodeResult:
    """Голая LLM без коуча. Пропускает EDA, торопится, средний скор."""
    steps: list[Step] = [
        Step(
            idx=0,
            stage=Stage.EDA,
            action=Action(
                type=ActionType.PLAN,
                content=(
                    "Бинарная классификация churn_small, метрика roc_auc. "
                    "План: быстро обучить logreg как бейзлайн, потом попробовать бустинг."
                ),
            ),
            result="plan noted",
            tokens_used=180,
        ),
        Step(
            idx=1,
            stage=Stage.BASELINE,
            action=Action(
                type=ActionType.CODE,
                content=(
                    "from sklearn.linear_model import LogisticRegression\n"
                    "from sklearn.model_selection import train_test_split\n"
                    "X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)\n"
                    "model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)\n"
                ),
            ),
            result="code written (12 lines)",
            tokens_used=420,
        ),
        Step(
            idx=2,
            stage=Stage.BASELINE,
            action=Action(type=ActionType.RUN, content="run current code"),
            result="val roc_auc=0.781",
            val_score=0.781,
            tokens_used=90,
        ),
        Step(
            idx=3,
            stage=Stage.IMPROVE,
            action=Action(
                type=ActionType.CODE,
                content=(
                    "from sklearn.ensemble import GradientBoostingClassifier\n"
                    "model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)\n"
                    "model.fit(X_tr, y_tr)\n"
                ),
            ),
            result="code written (18 lines)",
            tokens_used=510,
        ),
        Step(
            idx=4,
            stage=Stage.IMPROVE,
            action=Action(type=ActionType.RUN, content="run current code"),
            result="val roc_auc=0.812",
            val_score=0.812,
            tokens_used=95,
        ),
        Step(
            idx=5,
            stage=Stage.SUBMIT,
            action=Action(
                type=ActionType.SUBMIT,
                content="proba = model.predict_proba(X_test)[:, 1]; submit(proba)",
            ),
            result="submission accepted, test roc_auc=0.804",
            tokens_used=210,
        ),
    ]

    return EpisodeResult(
        task_id=TASK_ID,
        agent="baseline",
        seed=0,
        steps=steps,
        final_test_score=0.804,
        checklist_coverage=0.45,
        total_tokens=sum(s.tokens_used for s in steps),
        config={
            "model": "claude-opus-4-7",
            "agent_kind": "react",
            "max_steps": 12,
            "budget_tokens": 4000,
        },
    )


def build_scaffold() -> EpisodeResult:
    """Тот же таск, но с коучем и подсказками. Полный пайплайн, выше скор."""
    steps: list[Step] = [
        Step(
            idx=0,
            stage=Stage.EDA,
            action=Action(
                type=ActionType.PLAN,
                content=(
                    "Задача churn_small (бинарная классификация), метрика roc_auc. "
                    "Сначала зафиксирую честный val-split (стратификация по таргету), "
                    "потом EDA, потом бейзлайн, потом улучшения."
                ),
            ),
            result="plan noted",
            tokens_used=260,
            hints=[
                Hint(
                    stage=Stage.EDA,
                    item_id="val_split",
                    level=1,
                    text=(
                        "Подумай, как ты будешь сравнивать запуски между собой: "
                        "что должно быть зафиксировано в валидации, чтобы цифры были честными?"
                    ),
                ),
            ],
        ),
        Step(
            idx=1,
            stage=Stage.EDA,
            action=Action(
                type=ActionType.EDA,
                content=(
                    "df.info(); df.describe(); df.isna().mean(); "
                    "df['churn'].value_counts(normalize=True); "
                    "df.select_dtypes('object').nunique()"
                ),
            ),
            result=(
                "14 features (10 numeric, 4 categorical); "
                "missing: income 3.1%, last_login_days 0.4%; "
                "target prior P(churn=1)=0.27; income right-skewed (skew=2.1)"
            ),
            tokens_used=480,
        ),
        Step(
            idx=2,
            stage=Stage.BASELINE,
            action=Action(
                type=ActionType.CODE,
                content=(
                    "from sklearn.model_selection import StratifiedKFold\n"
                    "from sklearn.ensemble import GradientBoostingClassifier\n"
                    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)\n"
                    "model = GradientBoostingClassifier(random_state=0)\n"
                    "# cv roc_auc по 5 фолдам\n"
                ),
            ),
            result="code written (24 lines)",
            tokens_used=560,
        ),
        Step(
            idx=3,
            stage=Stage.BASELINE,
            action=Action(type=ActionType.RUN, content="run current code"),
            result="val roc_auc=0.794 ± 0.012 (5-fold)",
            val_score=0.794,
            tokens_used=110,
        ),
        Step(
            idx=4,
            stage=Stage.IMPROVE,
            action=Action(
                type=ActionType.CODE,
                content=(
                    "income_log = np.log1p(df['income'])\n"
                    "tenure_bucket = pd.cut(df['tenure_days'], bins=[0, 30, 180, 365, 9999])\n"
                    "# target encoding для cat-фичей считаю внутри CV-фолда, без утечки\n"
                ),
            ),
            result="code written (38 lines)",
            tokens_used=640,
            hints=[
                Hint(
                    stage=Stage.IMPROVE,
                    item_id="target_skew",
                    level=2,
                    text=(
                        "Ты заметил, как распределена 'income'? "
                        "Уверен, что твои фичи и модель устойчивы к таким хвостам, "
                        "или их стоит как-то подвинуть?"
                    ),
                ),
            ],
        ),
        Step(
            idx=5,
            stage=Stage.IMPROVE,
            action=Action(type=ActionType.RUN, content="run current code"),
            result="val roc_auc=0.836 ± 0.009 (5-fold)",
            val_score=0.836,
            tokens_used=120,
        ),
        Step(
            idx=6,
            stage=Stage.IMPROVE,
            action=Action(
                type=ActionType.CODE,
                content=(
                    "import lightgbm as lgb\n"
                    "model = lgb.LGBMClassifier(\n"
                    "    n_estimators=600, learning_rate=0.03,\n"
                    "    num_leaves=31, min_child_samples=20, random_state=0,\n"
                    ")\n"
                    "# подобрал n_estimators/learning_rate через быстрый 3-fold sweep\n"
                ),
            ),
            result="code written (46 lines)",
            tokens_used=700,
            hints=[
                Hint(
                    stage=Stage.IMPROVE,
                    item_id="model_family",
                    level=1,
                    text=(
                        "На таком объёме фичей и при таком распределении "
                        "стоит ли вообще оставаться на одной модели — или сравнить семейства?"
                    ),
                ),
            ],
        ),
        Step(
            idx=7,
            stage=Stage.IMPROVE,
            action=Action(type=ActionType.RUN, content="run current code"),
            result="val roc_auc=0.861 ± 0.008 (5-fold)",
            val_score=0.861,
            tokens_used=125,
        ),
        Step(
            idx=8,
            stage=Stage.SUBMIT,
            action=Action(
                type=ActionType.SUBMIT,
                content=(
                    "best = lgb.LGBMClassifier(...).fit(X, y)\n"
                    "proba = best.predict_proba(X_test)[:, 1]\n"
                    "submit(proba)"
                ),
            ),
            result="submission accepted, test roc_auc=0.852",
            tokens_used=260,
        ),
    ]

    return EpisodeResult(
        task_id=TASK_ID,
        agent="scaffold",
        seed=0,
        steps=steps,
        final_test_score=0.852,
        checklist_coverage=0.85,
        total_tokens=sum(s.tokens_used for s in steps),
        config={
            "model": "claude-opus-4-7",
            "agent_kind": "react+coach",
            "max_steps": 16,
            "budget_tokens": 8000,
            "hint_policy": "L1_first_then_L2_if_stuck",
        },
    )


def build_scaffold_seed1() -> EpisodeResult:
    """Второй scaffold-прогон на другом сиде для режима сравнения."""
    steps: list[Step] = [
        Step(
            idx=0,
            stage=Stage.EDA,
            action=Action(
                type=ActionType.PLAN,
                content=(
                    "Задача churn_small, метрика roc_auc. Сначала фиксирую split и "
                    "проверяю базовые риски утечек, затем baseline и цикл улучшений."
                ),
            ),
            result="plan noted",
            tokens_used=240,
            hints=[
                Hint(
                    stage=Stage.EDA,
                    item_id="val_split",
                    level=1,
                    text="Как будешь держать одинаковую валидацию между запусками?",
                ),
            ],
        ),
        Step(
            idx=1,
            stage=Stage.EDA,
            action=Action(
                type=ActionType.EDA,
                content="df.isna().mean(); df['churn'].value_counts(normalize=True); df.describe(include='all')",
            ),
            result="missing: income 3.0%, city 0.2%; class balance 27/73",
            tokens_used=430,
        ),
        Step(
            idx=2,
            stage=Stage.BASELINE,
            action=Action(
                type=ActionType.CODE,
                content=(
                    "from sklearn.linear_model import LogisticRegression\n"
                    "from sklearn.model_selection import StratifiedKFold\n"
                    "# ohe + logreg baseline\n"
                ),
            ),
            result="code written (22 lines)",
            tokens_used=520,
        ),
        Step(
            idx=3,
            stage=Stage.BASELINE,
            action=Action(type=ActionType.RUN, content="run current code"),
            result="val roc_auc=0.802 ± 0.011",
            val_score=0.802,
            tokens_used=105,
        ),
        Step(
            idx=4,
            stage=Stage.IMPROVE,
            action=Action(
                type=ActionType.CODE,
                content=(
                    "income_log = np.log1p(income)\n"
                    "# target encoding для high-cardinality категориальных признаков внутри CV\n"
                ),
            ),
            result="code written (31 lines)",
            tokens_used=610,
            hints=[
                Hint(
                    stage=Stage.IMPROVE,
                    item_id="feature_shift",
                    level=2,
                    text="Ты проверил, какие признаки реально дают прирост после кодирования?",
                ),
            ],
        ),
        Step(
            idx=5,
            stage=Stage.IMPROVE,
            action=Action(type=ActionType.RUN, content="run current code"),
            result="val roc_auc=0.844 ± 0.008",
            val_score=0.844,
            tokens_used=115,
        ),
        Step(
            idx=6,
            stage=Stage.SUBMIT,
            action=Action(
                type=ActionType.SUBMIT,
                content="proba = best_model.predict_proba(X_test)[:, 1]; submit(proba)",
            ),
            result="submission accepted, test roc_auc=0.839",
            tokens_used=220,
        ),
    ]

    return EpisodeResult(
        task_id=TASK_ID,
        agent="scaffold",
        seed=1,
        steps=steps,
        final_test_score=0.839,
        checklist_coverage=0.79,
        total_tokens=sum(s.tokens_used for s in steps),
        config={
            "model": "claude-opus-4-7",
            "agent_kind": "react+coach",
            "max_steps": 14,
            "budget_tokens": 7000,
            "hint_policy": "stage-scoped-escalation",
        },
    )


def _summary(ep: EpisodeResult) -> str:
    return (
        f"agent={ep.agent!r:>11}  steps={len(ep.steps):>2}  "
        f"final_test_score={ep.final_test_score}  total_tokens={ep.total_tokens}"
    )


def main() -> None:
    baseline = build_baseline()
    scaffold = build_scaffold()
    scaffold_seed1 = build_scaffold_seed1()

    BASELINE_PATH.write_text(baseline.to_json(), encoding="utf-8")
    SCAFFOLD_PATH.write_text(scaffold.to_json(), encoding="utf-8")
    SCAFFOLD_SEED1_PATH.write_text(scaffold_seed1.to_json(), encoding="utf-8")

    print(f"wrote {BASELINE_PATH.relative_to(_REPO_ROOT)}")
    print(f"wrote {SCAFFOLD_PATH.relative_to(_REPO_ROOT)}")
    print(f"wrote {SCAFFOLD_SEED1_PATH.relative_to(_REPO_ROOT)}")

    # Round-trip: читаем обратно через from_json и печатаем сводку.
    baseline_back = EpisodeResult.from_json(BASELINE_PATH.read_text(encoding="utf-8"))
    scaffold_back = EpisodeResult.from_json(SCAFFOLD_PATH.read_text(encoding="utf-8"))
    scaffold_seed1_back = EpisodeResult.from_json(SCAFFOLD_SEED1_PATH.read_text(encoding="utf-8"))

    print()
    print("re-parsed via EpisodeResult.from_json():")
    print(f"  {_summary(baseline_back)}")
    print(f"  {_summary(scaffold_back)}")
    print(f"  {_summary(scaffold_seed1_back)}")


if __name__ == "__main__":
    main()
