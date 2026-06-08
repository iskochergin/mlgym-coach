"""Ablation: какой уровень подсказок реально помогает.

3 задачи (titanic, california_housing, adult_income) × 4 режима × 3 сида = 36 эпизодов.
Режимы:
    baseline       — DummyCoach
    scaffold_L1    — Coach, max_hint_level=1 (только L1)
    scaffold_L1L2  — Coach, max_hint_level=2 (L1 + L2)
    scaffold_full  — Coach без отсечения (L1..L3)

Пишет эпизоды в runs/ablation_hints/<task>/seed<N>/<mode>/episode.json — дашборд
читает их как обычные.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.baseline import BaselineAgent
from core.types import Action, ActionType, EpisodeResult
from env.executor import reset_executor
from env.gym import Env
from env.runner_adapter import resolve_coach
from tasks.task_loader import load_task

TASKS = [
    "tasks/specs/titanic_survival.yaml",
    "tasks/specs/california_housing.yaml",
    "tasks/specs/adult_income.yaml",
]
MODES = [
    ("baseline", "dummy", 3),       # max_hint_level ничего не меняет (DummyCoach молчит)
    ("scaffold_L1", "real", 1),
    ("scaffold_L1L2", "real", 2),
    ("scaffold_full", "real", 3),
]
SEEDS = [0, 1, 2]
TOKEN_BUDGET = 30_000
MAX_STEPS = 18


def run_episode(spec_path: str, mode: str, coach_kind: str, max_hint_level: int, seed: int) -> dict:
    loaded = load_task(spec_path)
    task = loaded.task
    run_dir = _REPO_ROOT / "runs" / "ablation_hints" / task.id / f"seed{seed}" / mode
    coach = resolve_coach(coach_kind, max_hint_level=max_hint_level)
    reset_executor()
    env = Env(
        task=task,
        coach=coach,
        token_budget=TOKEN_BUDGET,
        seed=seed,
        agent_name=mode,
        hidden_labels_path=str(loaded.hidden_labels_path),
        run_dir=str(run_dir),
        max_steps=MAX_STEPS,
    )
    agent = BaselineAgent()
    obs = env.reset()
    for i in range(MAX_STEPS):
        action = agent.act(obs)
        obs = env.step(action, tokens_used=getattr(agent, "last_tokens", None))
        executed = obs.history[-1].action if obs.history else action
        if executed.type == ActionType.SUBMIT:
            break
        if obs.tokens_left <= 0:
            break
        # last-chance submit: получили val_score под конец — финализируем сами.
        last_val = obs.history[-1].val_score if obs.history else None
        last_run_ok = (
            executed.type == ActionType.RUN
            and last_val is not None
            and last_val == last_val  # NaN-проверка
        )
        if last_run_ok and (MAX_STEPS - (i + 1)) <= 1:
            obs = env.step(Action(type=ActionType.SUBMIT, content="[LOOP_FORCED_SUBMIT]"))
            break
    res = env.result()
    return dict(task=task.id, mode=mode, seed=seed, steps=len(res.steps),
                tokens=res.total_tokens, coverage=res.checklist_coverage,
                hints=sum(len(s.hints) for s in res.steps),
                final=res.final_test_score)


def main() -> None:
    t0 = time.time()
    rows = []
    total = len(TASKS) * len(MODES) * len(SEEDS)
    i = 0
    for spec in TASKS:
        for mode, coach_kind, max_hl in MODES:
            for seed in SEEDS:
                i += 1
                t = time.time()
                try:
                    row = run_episode(spec, mode, coach_kind, max_hl, seed)
                    print(f"  [{i:2d}/{total}] {row['task']:22s} {mode:14s} seed{seed} "
                          f"steps={row['steps']:2d} tok={row['tokens']:5d} cov={row['coverage']:.2f} "
                          f"hints={row['hints']:2d} final={row['final']} ({time.time()-t:.0f}s)", flush=True)
                    rows.append(row)
                except Exception as e:
                    import traceback
                    tid = Path(spec).stem.replace(".yaml", "")
                    print(f"  [{i:2d}/{total}] FAIL {tid} {mode} seed{seed}: {e!r}", flush=True)
                    traceback.print_exc()
                    # Плейсхолдер, чтобы дашборд и aggregate видели «упало», а не «ещё идёт».
                    rows.append(dict(task=tid, mode=mode, seed=seed, steps=0, tokens=None,
                                     coverage=None, hints=None, final=None,
                                     error=repr(e)[:200]))
    out = _REPO_ROOT / "runs" / "ablation_hints" / "ablation_summary.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nИТОГО: {len(rows)}/{total} эпизодов, wall-clock {time.time()-t0:.0f}s")
    print(f"  сводка → {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
