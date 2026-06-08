"""Параллельный прогон main_bvs_v2 (60) + ablation_hints (36) с ThreadPoolExecutor.

OpenAI держит >>500 RPM на mini-моделях, поэтому 4–6 параллельных эпизодов
проходят без rate limit. Каждый эпизод полностью независим — пишет в свой run_dir.

Запуск:
    MLGYM_LLM=openai python3 scripts/run_all_parallel.py [--workers 4] [--tag NAME]

--tag добавляет суффикс к корневому run_dir, чтобы сравнивать прогоны разных
моделей без перезаписи: runs/main_bvs_v2_<tag> и runs/ablation_hints_<tag>.
Модель берётся из OPENAI_MODEL (см. .env / llm_client).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.baseline import BaselineAgent
from core.types import Action, ActionType
from env.executor import reset_executor
from env.gym import Env
from env.runner_adapter import resolve_coach
from tasks.task_loader import load_task

# ─────────────────── Эксперименты ───────────────────

MAIN_TASKS = [
    "tasks/specs/breast_cancer_roc_auc.yaml",
    "tasks/specs/diabetes_rmse.yaml",
    "tasks/specs/titanic_survival.yaml",
    "tasks/specs/california_housing.yaml",
    "tasks/specs/adult_income.yaml",
    "tasks/specs/wine_quality.yaml",
    "tasks/specs/credit_g.yaml",
    "tasks/specs/bank_marketing.yaml",
]
MAIN_MODES = [
    # (agent_name, coach_kind, max_hint_level, run_dir_subpath)
    ("baseline", "dummy", 3, "baseline_dummy"),
    ("scaffold", "real", 3, "scaffold_real"),
]
MAIN_SEEDS = [0, 1, 2, 3, 4]

ABL_TASKS = [
    "tasks/specs/titanic_survival.yaml",
    "tasks/specs/california_housing.yaml",
    "tasks/specs/adult_income.yaml",
]
ABL_MODES = [
    ("baseline", "dummy", 3),
    ("scaffold_L1", "real", 1),
    ("scaffold_L1L2", "real", 2),
    ("scaffold_full", "real", 3),
]
ABL_SEEDS = [0, 1, 2]

TOKEN_BUDGET = int(os.environ.get("MLGYM_TOKEN_BUDGET", 30_000))
MAX_STEPS = int(os.environ.get("MLGYM_MAX_STEPS", 18))


# ─────────────────── Один эпизод ───────────────────


def _run_one(spec_path: str, agent_name: str, coach_kind: str, max_hint_level: int,
             seed: int, run_dir: Path) -> dict:
    loaded = load_task(spec_path)
    task = loaded.task
    coach = resolve_coach(coach_kind, max_hint_level=max_hint_level)
    reset_executor()
    env = Env(
        task=task,
        coach=coach,
        token_budget=TOKEN_BUDGET,
        seed=seed,
        agent_name=agent_name,
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
        last_val = obs.history[-1].val_score if obs.history else None
        if (executed.type == ActionType.RUN
                and last_val is not None and last_val == last_val
                and (MAX_STEPS - (i + 1)) <= 1):
            obs = env.step(Action(type=ActionType.SUBMIT, content="[LOOP_FORCED_SUBMIT]"))
            break
    res = env.result()
    return dict(
        task=task.id, agent=agent_name, mode=agent_name, seed=seed,
        steps=len(res.steps), tokens=res.total_tokens,
        coverage=res.checklist_coverage,
        hints=sum(len(s.hints) for s in res.steps),
        final=res.final_test_score,
    )


# ─────────────────── Оркестрация ───────────────────


def _main_root() -> Path:
    tag = os.environ.get("MLGYM_RUN_TAG", "")
    suffix = f"_{tag}" if tag else ""
    return _REPO_ROOT / "runs" / f"main_bvs_v2{suffix}"


def _ablation_root() -> Path:
    tag = os.environ.get("MLGYM_RUN_TAG", "")
    suffix = f"_{tag}" if tag else ""
    return _REPO_ROOT / "runs" / f"ablation_hints{suffix}"


def _job_main(task: str, mode: tuple, seed: int) -> dict:
    agent_name, coach_kind, max_hl, subpath = mode
    tid = load_task(task).task.id
    run_dir = _main_root() / subpath / tid / f"seed{seed}" / agent_name
    return _run_one(task, agent_name, coach_kind, max_hl, seed, run_dir)


def _job_ablation(task: str, mode: tuple, seed: int) -> dict:
    mode_name, coach_kind, max_hl = mode
    tid = load_task(task).task.id
    run_dir = _ablation_root() / tid / f"seed{seed}" / mode_name
    return _run_one(task, mode_name, coach_kind, max_hl, seed, run_dir)


def run_pool(jobs: list[tuple], workers: int, label: str) -> list[dict]:
    rows: list[dict] = []
    t0 = time.time()
    print(f"=== {label}: {len(jobs)} эпизодов, workers={workers} ===", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): args for fn, *args in jobs}
        done = 0
        for fut in as_completed(futures):
            args = futures[fut]
            done += 1
            try:
                row = fut.result()
                fs = "—" if row["final"] is None else f"{row['final']:.4f}"
                print(f"  [{done:2d}/{len(jobs)}] OK   {row['task'][:22]:22s} {row['mode']:14s} "
                      f"seed{row['seed']} steps={row['steps']:2d} tok={row['tokens']:5d} "
                      f"final={fs} ({time.time()-t0:.0f}s wall)", flush=True)
                rows.append(row)
            except Exception as e:
                print(f"  [{done:2d}/{len(jobs)}] FAIL {args}: {e!r}", flush=True)
                traceback.print_exc()
                rows.append(dict(task=str(args), mode="FAIL", seed=-1, steps=0,
                                 tokens=None, coverage=None, hints=None, final=None, error=repr(e)))
    print(f"=== {label} done: {len(rows)}/{len(jobs)} за {time.time()-t0:.0f}s wall ===\n", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-main", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--tag", default=os.environ.get("MLGYM_RUN_TAG", ""),
                    help="суффикс для папок: runs/main_bvs_v2_<tag>")
    args = ap.parse_args()
    if args.tag:
        os.environ["MLGYM_RUN_TAG"] = args.tag
    model = os.environ.get("OPENAI_MODEL", "?")
    print(f"[run_all_parallel] OPENAI_MODEL={model}, tag={args.tag!r}", flush=True)

    if not args.skip_main:
        main_jobs = [(_job_main, t, m, s)
                     for t in MAIN_TASKS for m in MAIN_MODES for s in MAIN_SEEDS]
        rows = run_pool(main_jobs, args.workers, "MAIN_BVS_V2")
        out = _main_root() / "summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not args.skip_ablation:
        abl_jobs = [(_job_ablation, t, m, s)
                    for t in ABL_TASKS for m in ABL_MODES for s in ABL_SEEDS]
        rows = run_pool(abl_jobs, args.workers, "ABLATION_HINTS")
        out = _ablation_root() / "summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
