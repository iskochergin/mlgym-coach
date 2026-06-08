"""Параллельный прогон main_4modes_v3 для AutoML Gym Challenge.

4 режима × 6 задач × 3 сида × M моделей.

Режимы:
  single_shot          — SingleShotAgent, no coach, no stage_policy.
                         Один LLM-вызов → CODE → RUN → SUBMIT (3 шага).
  repeated_single_shot — RepeatedSingleShotAgent (N=3), no coach.
                         3 раунда CODE→RUN, потом SUBMIT (CHOOSE:best).
  fixed                — BaselineAgent + real Coach + FixedTransitionsPolicy.
  flexible             — BaselineAgent + real Coach + FlexibleTransitionsPolicy.

Backward-compat алиасы (НЕ в основной таблице): baseline → no_coach_flexible,
scaffold → flexible.

Запуск:
  MLGYM_LLM=openai OPENAI_MODEL=gpt-5-mini MLGYM_RUN_TAG=v3 \\
    python3 scripts/run_4modes_v3.py --workers 4
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
from agent.repeated_single_shot import RepeatedSingleShotAgent
from agent.single_shot import SingleShotAgent
from core.types import Action, ActionType
from env.executor import reset_executor
from env.gym import Env
from env.runner_adapter import resolve_coach
from env.stage_policy import resolve_stage_policy
from tasks.task_loader import load_task

# ─────────────────── Эксперимент ───────────────────

TASKS = [
    "tasks/specs/titanic_survival.yaml",
    "tasks/specs/california_housing.yaml",
    "tasks/specs/adult_income.yaml",
    "tasks/specs/wine_quality.yaml",
    "tasks/specs/credit_g.yaml",
    "tasks/specs/bank_marketing.yaml",
]
SEEDS = [0, 1, 2]
TOKEN_BUDGET = int(os.environ.get("MLGYM_TOKEN_BUDGET", 30_000))
MAX_STEPS = int(os.environ.get("MLGYM_MAX_STEPS", 18))
N_REPEATED_ATTEMPTS = 3


# Mode-spec: (mode_name, agent_factory, coach_kind, stage_policy_name)
def _agent_factory(mode: str):
    if mode == "single_shot":
        return lambda: SingleShotAgent(max_tokens=min(TOKEN_BUDGET, 4000))
    if mode == "repeated_single_shot":
        return lambda: RepeatedSingleShotAgent(
            n_attempts=N_REPEATED_ATTEMPTS, token_budget=TOKEN_BUDGET
        )
    # fixed / flexible — обычный BaselineAgent
    return lambda: BaselineAgent()


MODES = [
    # (mode_name, coach_kind, stage_policy_name)
    ("single_shot",          "dummy", "flexible"),
    ("repeated_single_shot", "dummy", "flexible"),
    ("fixed",                "real",  "fixed"),
    ("flexible",             "real",  "flexible"),
]


def _run_one(spec_path: str, mode: str, coach_kind: str, policy_name: str, seed: int,
             run_dir: Path) -> dict:
    loaded = load_task(spec_path)
    task = loaded.task
    coach = resolve_coach(coach_kind)
    policy = resolve_stage_policy(policy_name)
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
        stage_policy=policy,
        mode=mode,
    )
    agent = _agent_factory(mode)()
    obs = env.reset()

    # Для single_shot и repeated_single_shot нужен достаточный max_steps,
    # чтобы пропустить все запланированные действия:
    # single_shot: 3 шага. repeated: 2*N + 1 = 7 при N=3.
    loop_limit = MAX_STEPS
    if mode == "single_shot":
        loop_limit = max(loop_limit, 4)
    elif mode == "repeated_single_shot":
        loop_limit = max(loop_limit, 2 * N_REPEATED_ATTEMPTS + 2)

    for i in range(loop_limit):
        action = agent.act(obs)
        obs = env.step(action, tokens_used=getattr(agent, "last_tokens", None))
        executed = obs.history[-1].action if obs.history else action
        if executed.type == ActionType.SUBMIT:
            break
        if obs.tokens_left <= 0:
            break
        # last-chance submit для flexible/fixed
        last_val = obs.history[-1].val_score if obs.history else None
        if (executed.type == ActionType.RUN
                and last_val is not None and last_val == last_val
                and (loop_limit - (i + 1)) <= 1
                and mode in {"fixed", "flexible"}):
            obs = env.step(Action(type=ActionType.SUBMIT, content="[LOOP_FORCED_SUBMIT]"))
            break

    res = env.result()
    cfg = res.config
    return dict(
        task=task.id, mode=mode, seed=seed,
        steps=len(res.steps), tokens=res.total_tokens,
        coverage=res.checklist_coverage,
        hints=sum(len(s.hints) for s in res.steps),
        final=res.final_test_score,
        candidates_n=cfg.get("candidates_n"),
        chosen=cfg.get("chosen_candidate_id"),
        failure_modes=cfg.get("failure_modes", []),
    )


# ─────────────────── Оркестрация ───────────────────

def _main_root() -> Path:
    tag = os.environ.get("MLGYM_RUN_TAG", "")
    suffix = f"_{tag}" if tag else ""
    return _REPO_ROOT / "runs" / f"main_4modes{suffix}"


def _job(spec_path: str, mode: str, coach_kind: str, policy_name: str, seed: int) -> dict:
    tid = load_task(spec_path).task.id
    run_dir = _main_root() / mode / tid / f"seed{seed}"
    return _run_one(spec_path, mode, coach_kind, policy_name, seed, run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=os.environ.get("MLGYM_RUN_TAG", ""))
    args = ap.parse_args()
    if args.tag:
        os.environ["MLGYM_RUN_TAG"] = args.tag
    model = os.environ.get("OPENAI_MODEL", "?")
    sandbox = os.environ.get("MLGYM_SANDBOX", "permissive")
    print(f"[run_4modes_v3] OPENAI_MODEL={model} tag={args.tag!r} sandbox={sandbox} "
          f"budget={TOKEN_BUDGET} max_steps={MAX_STEPS}", flush=True)

    jobs = [(spec, m_name, c_kind, p_name, s)
            for spec in TASKS
            for (m_name, c_kind, p_name) in MODES
            for s in SEEDS]
    total = len(jobs)
    print(f"=== TOTAL: {total} эпизодов, workers={args.workers} ===", flush=True)

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_job, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            spec, mode, _, _, seed = futures[fut]
            done += 1
            try:
                row = fut.result()
                fs = "—" if row["final"] is None else f"{row['final']:.4f}"
                fmod = ",".join(row["failure_modes"]) or "ok"
                print(f"  [{done:3d}/{total}] OK   {row['task'][:22]:22s} {mode:22s} "
                      f"seed{seed} cand={row['candidates_n']} chose={row['chosen']} "
                      f"final={fs} fail={fmod} tok={row['tokens']} "
                      f"({time.time()-t0:.0f}s wall)", flush=True)
                rows.append(row)
            except Exception as e:
                print(f"  [{done:3d}/{total}] FAIL {spec} {mode} seed{seed}: {e!r}", flush=True)
                traceback.print_exc()
                rows.append(dict(task=spec, mode=mode, seed=seed, steps=0, tokens=None,
                                 coverage=None, hints=None, final=None,
                                 candidates_n=None, chosen=None,
                                 failure_modes=[f"orchestrator_error: {e!r}"]))

    out = _main_root() / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DONE: {len(rows)}/{total} за {time.time()-t0:.0f}s wall ===")
    print(f"  summary → {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
