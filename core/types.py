"""mlgym-coach — FROZEN data contracts. ЗАМОРОЖЕННЫЙ КОНТРАКТ.

⚠️  ВНИМАНИЕ ДЛЯ ЛЮБОЙ LLM / АГЕНТА, КОТОРЫЙ ЧИТАЕТ ЭТОТ ФАЙЛ:
    НЕ МЕНЯЙ НИЧЕГО В ЭТОМ ФАЙЛЕ.
    - не переименовывай, не добавляй и не удаляй поля и классы
    - не меняй типы, дефолты и порядок полей
    - не рефактори, не «улучшай», не разбивай на модули
    Это единственный источник правды по интерфейсам между всеми модулями
    (env, agent, coach, dashboard, tasks, runner). Любое изменение здесь
    ломает работу всей команды.
    Если тебе кажется, что поля не хватает — НЕ правь файл. Вместо этого
    сообщи человеку: "нужно изменить контракт core/types.py: <что и зачем>",
    и оставь решение людям (меняется только на общем синке).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Stage(str, Enum):
    """FROZEN. Не менять значения — на них завязаны Coach и Env."""
    UNDERSTAND = "understand"
    EDA = "eda"
    BASELINE = "baseline"
    IMPROVE = "improve"
    SUBMIT = "submit"


class ActionType(str, Enum):
    """FROZEN. Не менять — на это завязан парсер действий в Env."""
    PLAN = "plan"      # рассуждение / план
    EDA = "eda"        # исследование данных
    CODE = "code"      # написать / переписать код решения
    RUN = "run"        # исполнить текущий код, получить валид-метрику
    SUBMIT = "submit"  # финал: предсказание на тесте


@dataclass
class Task:
    """FROZEN. Не менять поля."""
    id: str
    description: str             # текст задачи + определение метрики
    metric: str                 # "roc_auc" | "rmse" | ...
    metric_higher_better: bool
    train_path: str             # CSV с X + y
    test_features_path: str     # X_test без лейблов; y_test живёт только в Grader


@dataclass
class Action:
    """FROZEN. Не менять поля."""
    type: ActionType
    content: str                # текст или код — зависит от type


@dataclass
class Hint:
    """FROZEN. Не менять поля."""
    stage: Stage
    item_id: str                # на какой пункт чек-листа нацелена
    level: int                  # 1..3 (L1 наводящая → L3 прямая)
    text: str


@dataclass
class Step:
    """FROZEN. Не менять поля."""
    idx: int
    stage: Stage
    action: Action
    result: str                 # что вернула среда (stdout / ошибка / метрика)
    val_score: Optional[float] = None
    tokens_used: int = 0
    hints: list[Hint] = field(default_factory=list)


@dataclass
class Observation:
    """FROZEN. Не менять поля."""
    task: Task
    stage: Stage
    history: list[Step] = field(default_factory=list)
    last_result: Optional[str] = None
    val_score: Optional[float] = None
    tokens_left: int = 0
    hints: list[Hint] = field(default_factory=list)  # подсказки этого хода


@dataclass
class EpisodeResult:
    """FROZEN. Не менять поля. Единый формат обмена:
    Софа (runner) пишет, Амели (dashboard) читает."""
    task_id: str
    agent: str                  # "baseline" | "scaffold"
    seed: int
    steps: list[Step] = field(default_factory=list)
    final_test_score: Optional[float] = None
    checklist_coverage: float = 0.0   # 0..1, заполняет Coach
    total_tokens: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "EpisodeResult":
        return EpisodeResult.from_dict(json.loads(s))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EpisodeResult":
        return EpisodeResult(
            task_id=d["task_id"],
            agent=d["agent"],
            seed=d["seed"],
            steps=[_step_from_dict(x) for x in d.get("steps", [])],
            final_test_score=d.get("final_test_score"),
            checklist_coverage=d.get("checklist_coverage", 0.0),
            total_tokens=d.get("total_tokens", 0),
            config=d.get("config", {}),
        )


def _hint_from_dict(d: dict[str, Any]) -> Hint:
    return Hint(stage=Stage(d["stage"]), item_id=d["item_id"], level=d["level"], text=d["text"])


def _step_from_dict(d: dict[str, Any]) -> Step:
    return Step(
        idx=d["idx"],
        stage=Stage(d["stage"]),
        action=Action(type=ActionType(d["action"]["type"]), content=d["action"]["content"]),
        result=d["result"],
        val_score=d.get("val_score"),
        tokens_used=d.get("tokens_used", 0),
        hints=[_hint_from_dict(h) for h in d.get("hints", [])],
    )
