# mlgym-coach

> A gym for benchmarking how much a structured pipeline + Socratic hints improve LLM agents on tabular ML tasks — baseline vs scaffold, under a fixed token budget.

This file is the **full project context**. You can hand it as-is to any AI agent (Claude, etc.) so it understands what we are building, how the architecture works, which interfaces it may touch, and which area it is currently helping with. If you are an agent — read the whole file first, then the **"Instructions for AI agents"** section at the end.

---

## Idea

We are testing one hypothesis: **how much does a structured pipeline + guiding hints improve an LLM agent on classic tabular ML tasks** compared to a bare LLM.

- We take an LLM and an environment (`env`) in which the agent solves a tabular ML task.
- The agent acts step by step: `LLM → action → env → feedback`.
- Inside the environment there is a **hidden checklist** of the ideal pipeline. The agent cannot see it; the environment uses it to give guiding hints.
- A hard **token limit** per episode. The agent sees everything except `y_test`.
- The outcome is a comparison of two configurations: `baseline` (bare LLM) vs `scaffold` (LLM + pipeline + hints), with a fixed model and dataset, across several seeds.

Hints are **indirect**: not "remove the outliers in column X", but nudges that let the agent figure it out on its own. How direct they are is an experimental variable (see the L1–L3 ladder below).

---

## How it works (the loop)

```
        ┌─────────── Observation (state, stage, hints, tokens_left) ───────────┐
        │                                                                      │
        ▼                                                                      │
   ┌─────────┐        Action         ┌─────────────────────────────┐          │
   │   LLM   │ ───────────────────▶  │            Env              │ ─────────┘
   │ (agent) │                       │  parser · sandbox · budget  │
   └─────────┘                       │            │                │
                                     │            ▼                │
                                     │   ┌──────────────────┐      │
                                     │   │  Coach (hidden)  │      │
                                     │   │ checklist+hints  │      │
                                     │   └──────────────────┘      │
                                     └─────────────────────────────┘
                                                  │ submit
                                                  ▼
                                           Grader → final test score
```

Data access rules:
- The agent sees: `train (X+y)`, the task description, the metric, the test features (`X_test`), and hints.
- The agent **does not see**: `y_test` or the grader code. The test metric is computed only on `submit`.
- We catch leaks: overfitting to validation, attempts to pull labels out of train.

---

## Ideal pipeline (v1)

Linear, 5 stages, one loop. It is both the agent's "track" and the skeleton of the Coach checklist (one group of items per stage).

```
1 · EDA               →  distributions, missing values, types, leaks
2 · Baseline          →  simplest model, first validation score
3 · Improve (loop)    →  features · model · tuning — while the score keeps going up
4 · Submit            →  predict on test, final score
```

For each checklist item the Coach keeps a **detector** (has the agent completed the step?) and a **hint ladder**:

| Level | What we give (example: "check the target distribution") |
|---|---|
| L0 | stay silent (default) |
| L1 — direction | "take a look at how your target variable is structured — it affects the choice of loss" |
| L2 — question | "have you looked at the skewness of the distributions? are you sure your metric is robust to it?" |
| L3 — direct | "log-transform the target" (normally we don't get this far) |

A hint is given **only for uncovered items** and escalates in level if the agent is stuck for N steps.

This is an intentionally simple version: no branching, no separate "ensemble/calibration" stage (for now it lives inside "Improve"), a single loop. Once the basic version works, stage 4 can be split up.

---

## Repository layout

One folder per owner: boundaries in git match areas of responsibility → almost no conflicts.

```
mlgym-coach/
  core/        # CONTRACTS: Task, Action, Observation, Stage, Hint, Step, EpisodeResult
               #   types.py — FROZEN; changed only via a sync
  env/         # [owner: Vanya] gym loop, action parser, sandbox, token budget
               #   gym.py · executor.py (v1 stub) · dummy_coach.py · run_slice.py (demo)
  agent/       # [owner: Vanya] baseline (ReAct) + scaffold (tree-search)
               #   scripted.py — stub agent without an LLM, for testing the loop
  coach/       # [owner: Egor] checklist, detectors, L1–L3 hints, grader, metrics
  dashboard/   # [owner: Amelie] Streamlit viewer of runs on top of EpisodeResult.json
  tasks/       # [owner: Sofia] datasets + task specs (train + hidden test + metric)
  runner/      # [owner: Sofia] experiment harness (fixes LLM + dataset, N seeds)
  reports/     # [owner: Sofia] aggregated results, baseline vs scaffold tables

  examples/    # reference EpisodeResult fixtures (episode_baseline.json, episode_scaffold.json)
               #   + make_examples.py — generator via core.types. Input for Amelie/Sofia.
  runs/        # run artifacts (slice_demo.json, etc.). In .gitignore, not committed.
```

The only shared editing area is `core/`. It is frozen at the start and changed only via a quick sync.

The repo currently contains a **thin end-to-end slice**: `env/` + `agent/scripted.py` run the loop end-to-end on stubs (`executor.py` returns fake scores, `dummy_coach.py` stays silent). Run the demo with `python env/run_slice.py` — it writes the result to `runs/slice_demo.json`. The format is guaranteed to match the `core/types.py` contract.

---

## `core/` contracts (the interface everyone codes against)

This is draft v1. All modules import from here and **do not change fields without a sync**.

```python
# core/types.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Stage(str, Enum):
    UNDERSTAND = "understand"
    EDA        = "eda"
    BASELINE   = "baseline"
    IMPROVE    = "improve"
    SUBMIT     = "submit"

class ActionType(str, Enum):
    PLAN   = "plan"     # reasoning / plan
    EDA    = "eda"      # data exploration
    CODE   = "code"     # write / rewrite the solution code
    RUN    = "run"      # execute the current code, get the validation metric
    SUBMIT = "submit"   # final: predict on test

@dataclass
class Task:
    id: str
    description: str            # task text + metric definition
    metric: str                 # "roc_auc" | "rmse" | ...
    metric_higher_better: bool
    train_path: str             # CSV with X + y
    test_features_path: str     # X_test without labels
    # y_test lives ONLY in the Grader, not available to the agent

@dataclass
class Action:
    type: ActionType
    content: str                # text or code — depends on type

@dataclass
class Hint:
    stage: Stage
    item_id: str                # which checklist item it targets
    level: int                  # 1..3 (L1 nudge → L3 direct)
    text: str

@dataclass
class Step:
    idx: int
    stage: Stage
    action: Action
    result: str                 # what the env returned (stdout / error / metric)
    val_score: Optional[float]
    tokens_used: int
    hints: list[Hint] = field(default_factory=list)

@dataclass
class Observation:
    task: Task
    stage: Stage                # the agent's current stage
    history: list[Step]
    last_result: Optional[str]
    val_score: Optional[float]
    tokens_left: int
    hints: list[Hint] = field(default_factory=list)   # hints for this turn

@dataclass
class EpisodeResult:
    task_id: str
    agent: str                  # "baseline" | "scaffold"
    seed: int
    steps: list[Step]
    final_test_score: Optional[float]
    checklist_coverage: float   # 0..1, filled in by the Coach
    total_tokens: int
    config: dict = field(default_factory=dict)
```

Key module signatures:

```python
# env/gym.py
class Env:
    def reset(self, task: Task) -> Observation: ...
    def step(self, action: Action) -> Observation: ...   # advances state, calls the Coach
    def result(self) -> EpisodeResult: ...

# agent/base.py
class Agent:
    def act(self, obs: Observation) -> Action: ...

# coach/coach.py
class Coach:
    def assess(self, obs: Observation) -> tuple[float, list[Hint]]:
        """Returns the current checklist coverage (0..1) and the hints for this turn."""

# coach/grader.py
def score_submission(task: Task, predictions) -> float: ...   # computes the metric on y_test
```

Glue points: `Observation` (Env→Agent, read by the Coach), `Hint` (Coach→Env→Agent), `EpisodeResult` (everything → dashboard + runner).

---

**Vanya — Agent + Env** (`agent/`, `env/`)
Vanya leads `agent/` + `env/` and owns `core/`. The other three have independent folders.

**Egor — Coach** (`coach/`) ![Team-lead](https://img.shields.io/badge/Team--lead-blue)
Writes the hidden 5-stage checklist (each stage is a set of items); detectors that use the `Observation` to determine which items the agent has completed; an L1–L3 hint ladder for each item with a delivery policy (only uncompleted items, escalate if the agent is stuck); and coverage metrics + hint statistics. The grader (`score_submission`) lives here too.
*Mocks: `Observation` and a couple of fake trajectories — no real environment needed.*

**Amelie — Interface** (`dashboard/`)
Dashboard (Streamlit on top of `EpisodeResult.json`): list of runs, timeline of the agent's actions (stage → action → turn result), progress across the 5 stages with the hints given, token/score charts per step, comparison of two runs (baseline vs scaffold).
*Mocks: one or two hand-written `EpisodeResult.json` examples — works fully standalone, without anyone else's code.*

**Sofia — Runner + datasets** (`tasks/`, `runner/`, `reports/`)
2–3 simple tabular datasets (train + hidden test + metric function + `Task` spec) with protection against `y_test` leaks; a runner that fixes the LLM and dataset and runs the experiment over N seeds via a YAML config; an aggregator (mean ± SE, baseline vs scaffold table) that exports to `reports/`.
*Mocks: a fake env that returns a random score until the real one is ready.*

---

## Work plan

1. **Day zero (Vanya, ~1 hour):** freeze `core/types.py`. Everyone imports it and mocks the rest.
2. **Thin end-to-end slice (together, a couple of days):** 1 dataset, baseline agent, a dumb Coach without hints, env runs, runner produces an `EpisodeResult`, dashboard displays it. The goal is to wire all modules together once and catch interface bugs.
3. **In parallel:** everyone polishes their own folder against the contracts.
4. **Experiment (together):** baseline vs scaffold + ablation by layer, ≥3 seeds, mean ± SE → `reports/`.

---

## Evaluation

- **Outcome:** score on the held-out test; share of tasks with ≥X% improvement over a fixed baseline.
- **Process:** checklist coverage (how many steps the agent completed on its own), how many hints were given and at what level.
- **Efficiency:** tokens / cost to reach a given score; score at a fixed budget.
- **Statistics:** ≥3 seeds per configuration (agents are highly variable), mean ± SE.
- **Anti-cheating:** gains must come from model quality, not from leaks or overfitting to validation.

---

## Stack / conventions

- Python 3.11+, type hints, `dataclasses`. No heavy frameworks in v1.
- Lint/format: `ruff` + `black`.
- LLM client — model-neutral (OpenAI-compatible API), the model is set in the config.
- Sandbox: the agent's code runs in an isolated process/container.

---

## Instructions for AI agents

If this file was given to you so you can help one of the team members:

1. **Identify your area.** Ask (or infer from the task) which folder you are working in: `coach/`, `dashboard/`, or `tasks/`+`runner/`. Do not touch other people's folders.
2. **Code against `core/`.** Import types from `core/types.py` and do not change their fields. If a field is genuinely missing — don't edit it silently; raise it as a question for the contracts sync.
3. **Mock everything else.** You don't need anyone else's finished code: the Coach mocks `Observation`, the Interface mocks `EpisodeResult.json`, the Runner mocks a fake env. That is the whole point of working in parallel.
4. **Keep v1 simple.** Five stages, one loop, simple detectors (rules over code/history; LLM-judge comes later). Don't add stages or features that aren't in this file unless explicitly asked.
5. **Don't break the data rules.** Never give the agent access to `y_test` or the grader code. Any code that reads test labels lives only inside `coach/grader.py`.
6. **Treat this README** as the source of truth for architecture, contracts, and areas of responsibility.

---

## Running with Docker (just run everything)

One container: Streamlit dashboard + runner + agent.

1. Put your OpenAI key in `.env` at the repo root (see `SECRETS.md`):
   ```
   OPENAI_API_KEY=sk-...        # your key; sk-REPLACE-ME → MockLLM is used
   OPENAI_MODEL=gpt-5-mini
   MLGYM_LLM=openai             # mock | openai
   ```
   `.env` is not committed to git and does NOT end up in the image — the key is passed
   into the container at runtime via `env_file` in `docker-compose.yml`.

2. Start it:
   ```bash
   docker compose up --build
   ```

3. Open the dashboard: **http://localhost:8501**

Data survives image rebuilds — `runs/` and `tasks/uploads/` are mounted as
volumes. The code execution sandbox in this setup is NOT isolated (the agent writes our own
baseline code); strict isolation is a task for later, once third-party code is being executed.
