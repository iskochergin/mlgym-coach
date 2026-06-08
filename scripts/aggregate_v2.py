"""Агрегация main_bvs_v2 + ablation_hints в markdown-таблицы.

Использование:
    python -m scripts.aggregate_v2  # печатает таблицы и сохраняет в REPORT_v2_tables.md

Считает n_ok (число эпизодов с непустым final_test_score), mean ± SE для final,
средние токены, coverage и hints в группах (task, agent).
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RUNS = _REPO_ROOT / "runs"


def _episodes(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for p in sorted(root.rglob("episode.json")):
        try:
            ep = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(dict(
            path=str(p.relative_to(_REPO_ROOT)),
            task=ep.get("task_id"),
            agent=ep.get("agent"),
            seed=ep.get("seed"),
            steps=len(ep.get("steps", [])),
            tokens=ep.get("total_tokens"),
            coverage=ep.get("checklist_coverage"),
            hints=sum(len(s.get("hints", [])) for s in ep.get("steps", [])),
            final=ep.get("final_test_score"),
        ))
    return rows


def _mean_se(xs: list[float]) -> tuple[float, float, int]:
    """Среднее ± SE. SE через выборочное std (n−1), NaN при n≤1."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    m = st.mean(xs)
    se = (st.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
    return m, se, len(xs)


def _fmt(x: float, digits: int = 4) -> str:
    return "—" if x != x else f"{x:.{digits}f}"


def aggregate(name: str, rows: list[dict], group_keys: tuple[str, ...]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)
    header = "| " + " | ".join([*group_keys, "n_ok/n", "mean_final ± SE", "mean_tokens", "mean_coverage", "mean_hints"]) + " |"
    sep = "|" + "|".join("---" for _ in range(len(group_keys) + 5)) + "|"
    lines = [f"### {name}", "", header, sep]
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        finals = [r["final"] for r in rs]
        mf, se, n_ok = _mean_se(finals)
        # Усреднение только по эпизодам с реальным значением — иначе кэши/фейлы
        # тянут среднее вниз и режим с большей долей крашей кажется «экономнее».
        tok_vals = [r["tokens"] for r in rs if r["tokens"] is not None]
        cov_vals = [r["coverage"] for r in rs if r["coverage"] is not None]
        hint_vals = [r["hints"] for r in rs if r["hints"] is not None]
        mt = sum(tok_vals) / len(tok_vals) if tok_vals else float("nan")
        mc = sum(cov_vals) / len(cov_vals) if cov_vals else float("nan")
        mh = sum(hint_vals) / len(hint_vals) if hint_vals else float("nan")
        cells = ["—" if k is None else str(k) for k in key]
        cells += [f"{n_ok}/{len(rs)}",
                  f"{_fmt(mf)} ± {_fmt(se)}",
                  ("—" if mt != mt else f"{int(mt):,}"),
                  ("—" if mc != mc else f"{mc:.3f}"),
                  ("—" if mh != mh else f"{mh:.1f}")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    out = []

    main_rows = _episodes(RUNS / "main_bvs_v2")
    out.append(aggregate("main_bvs_v2 — по (task, agent)", main_rows, ("task", "agent")))

    abl_rows = _episodes(RUNS / "ablation_hints")
    out.append(aggregate("ablation_hints — по (task, agent)", abl_rows, ("task", "agent")))

    md = "\n\n".join(out)
    print(md)
    (_REPO_ROOT / "REPORT_v2_tables.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
