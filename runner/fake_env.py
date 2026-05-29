from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from core.types import Action, ActionType, EpisodeResult, Hint, Stage, Step, Task


@dataclass(frozen=True)
class FakeEnvConfig:
    model: str
    token_budget: int
    max_steps: int
    env_name: str = "fake"


class FakeEnv:
    """Deterministic runner stub that produces EpisodeResult without a real ML env."""

    def __init__(self, task: Task, agent: str, seed: int, config: FakeEnvConfig) -> None:
        self.task = task
        self.agent = agent
        self.seed = seed
        self.config = config
        self._rng = random.Random(_stable_seed(task.id, agent, seed))

    def run(self) -> EpisodeResult:
        val_scores, final_score = self._scores()
        coverage = self._coverage()
        steps = self._steps(val_scores, final_score)

        return EpisodeResult(
            task_id=self.task.id,
            agent=self.agent,
            seed=self.seed,
            steps=steps,
            final_test_score=final_score,
            checklist_coverage=coverage,
            total_tokens=sum(step.tokens_used for step in steps),
            config=self._episode_config(),
        )

    def _scores(self) -> tuple[list[float], float]:
        if self.task.metric_higher_better:
            base = 0.735 if self.agent == "baseline" else 0.785
            first = _clamp(base + self._rng.gauss(0, 0.018), 0.5, 0.99)
            second = _clamp(first + self._rng.uniform(0.015, 0.055), 0.5, 0.99)
            test = _clamp(second + self._rng.gauss(0, 0.012), 0.5, 0.99)
        else:
            base = 61.0 if self.agent == "baseline" else 55.0
            first = max(1.0, base + self._rng.gauss(0, 2.5))
            second = max(1.0, first - self._rng.uniform(1.0, 5.0))
            test = max(1.0, second + self._rng.gauss(0, 1.7))
        return [round(first, 4), round(second, 4)], round(test, 4)

    def _coverage(self) -> float:
        if self.agent == "scaffold":
            return round(_clamp(0.78 + self._rng.gauss(0, 0.04), 0.0, 1.0), 3)
        return round(_clamp(0.43 + self._rng.gauss(0, 0.05), 0.0, 1.0), 3)

    def _steps(self, val_scores: list[float], final_score: float) -> list[Step]:
        metric = self.task.metric
        steps = [
            Step(
                idx=0,
                stage=Stage.EDA,
                action=Action(
                    type=ActionType.PLAN,
                    content=f"Understand {self.task.id}, metric {metric}, then build baseline.",
                ),
                result="plan noted",
                tokens_used=self._tokens(150, 260),
                hints=self._hints(Stage.EDA, "validation_plan"),
            ),
            Step(
                idx=1,
                stage=Stage.EDA if self.agent == "scaffold" else Stage.BASELINE,
                action=Action(
                    type=ActionType.EDA if self.agent == "scaffold" else ActionType.CODE,
                    content=self._eda_or_baseline_content(),
                ),
                result=self._eda_or_baseline_result(),
                tokens_used=self._tokens(300, 560),
                hints=(
                    self._hints(Stage.EDA, "target_and_schema")
                    if self.agent == "scaffold"
                    else []
                ),
            ),
            Step(
                idx=2,
                stage=Stage.BASELINE,
                action=Action(type=ActionType.RUN, content="run current baseline"),
                result=f"val {metric}={val_scores[0]:.4f}",
                val_score=val_scores[0],
                tokens_used=self._tokens(70, 130),
            ),
            Step(
                idx=3,
                stage=Stage.IMPROVE,
                action=Action(
                    type=ActionType.CODE,
                    content="Try one stronger model or feature-processing variant.",
                ),
                result="code accepted (fake env)",
                tokens_used=self._tokens(360, 680),
                hints=self._hints(Stage.IMPROVE, "model_comparison"),
            ),
            Step(
                idx=4,
                stage=Stage.IMPROVE,
                action=Action(type=ActionType.RUN, content="run improved solution"),
                result=f"val {metric}={val_scores[1]:.4f}",
                val_score=val_scores[1],
                tokens_used=self._tokens(75, 135),
            ),
            Step(
                idx=5,
                stage=Stage.SUBMIT,
                action=Action(
                    type=ActionType.SUBMIT,
                    content="submit predictions for hidden test set",
                ),
                result=f"submission accepted, test {metric}={final_score:.4f}",
                tokens_used=self._tokens(150, 260),
            ),
        ]
        return steps[: self.config.max_steps]

    def _eda_or_baseline_content(self) -> str:
        if self.agent == "scaffold":
            return "Inspect schema, missing values, target distribution, and metric direction."
        return "Fit a simple baseline model with a fixed validation split."

    def _eda_or_baseline_result(self) -> str:
        if self.agent == "scaffold":
            return "eda noted: schema, missingness, and target summary checked"
        return "code accepted (fake env)"

    def _hints(self, stage: Stage, item_id: str) -> list[Hint]:
        if self.agent != "scaffold":
            return []
        return [
            Hint(
                stage=stage,
                item_id=item_id,
                level=1,
                text="Check the comparison protocol before trusting the next score.",
            )
        ]

    def _tokens(self, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def _episode_config(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "agent_kind": self.agent,
            "env": self.config.env_name,
            "max_steps": self.config.max_steps,
            "budget_tokens": self.config.token_budget,
            "metric": self.task.metric,
            "metric_higher_better": self.task.metric_higher_better,
        }


def _stable_seed(task_id: str, agent: str, seed: int) -> int:
    digest = hashlib.sha256(f"{task_id}:{agent}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
