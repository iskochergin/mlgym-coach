from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.types import EpisodeResult


@dataclass(frozen=True)
class ScoreSummary:
    mean: float
    se: float
    n: int


def main() -> None:
    args = _parse_args()
    csv_path, md_path = aggregate(args.input, args.output)
    print(csv_path.relative_to(_REPO_ROOT))
    print(md_path.relative_to(_REPO_ROOT))


def aggregate(input_dir: Path, output_base: Path) -> tuple[Path, Path]:
    input_dir = _resolve_repo_path(input_dir)
    output_base = _resolve_repo_path(output_base)

    episodes = _read_episodes(input_dir)
    rows = _summary_rows(episodes)

    csv_path = output_base.with_suffix(".csv")
    md_path = output_base.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    return csv_path, md_path


def _read_episodes(input_dir: Path) -> list[EpisodeResult]:
    paths = sorted(input_dir.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No EpisodeResult JSON files found under {input_dir}")
    return [EpisodeResult.from_json(path.read_text(encoding="utf-8")) for path in paths]


def _summary_rows(episodes: list[EpisodeResult]) -> list[dict[str, object]]:
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[str, dict[str, object]] = {}

    for episode in episodes:
        if episode.final_test_score is None:
            continue
        scores[(episode.task_id, episode.agent)].append(float(episode.final_test_score))
        metadata.setdefault(
            episode.task_id,
            {
                "metric": episode.config.get("metric", ""),
                "metric_higher_better": episode.config.get("metric_higher_better", True),
            },
        )

    rows: list[dict[str, object]] = []
    for task_id in sorted(metadata):
        task_meta = metadata[task_id]
        baseline = _summarize(scores.get((task_id, "baseline"), []))
        scaffold = _summarize(scores.get((task_id, "scaffold"), []))
        higher_better = bool(task_meta["metric_higher_better"])
        improvement = _improvement(baseline, scaffold, higher_better)

        rows.append(
            {
                "task_id": task_id,
                "metric": task_meta["metric"],
                "higher_better": higher_better,
                "baseline": _format_summary(baseline),
                "scaffold": _format_summary(scaffold),
                "baseline_mean": _mean_or_blank(baseline),
                "baseline_se": _se_or_blank(baseline),
                "baseline_n": baseline.n if baseline else 0,
                "scaffold_mean": _mean_or_blank(scaffold),
                "scaffold_se": _se_or_blank(scaffold),
                "scaffold_n": scaffold.n if scaffold else 0,
                "improvement": "" if improvement is None else round(improvement, 4),
            }
        )
    return rows


def _summarize(values: list[float]) -> ScoreSummary | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return ScoreSummary(mean=mean, se=se, n=len(values))


def _improvement(
    baseline: ScoreSummary | None,
    scaffold: ScoreSummary | None,
    higher_better: bool,
) -> float | None:
    if baseline is None or scaffold is None:
        return None
    if higher_better:
        return scaffold.mean - baseline.mean
    return baseline.mean - scaffold.mean


def _format_summary(summary: ScoreSummary | None) -> str:
    if summary is None:
        return ""
    return f"{summary.mean:.4f} +/- {summary.se:.4f}"


def _mean_or_blank(summary: ScoreSummary | None) -> float | str:
    return "" if summary is None else round(summary.mean, 4)


def _se_or_blank(summary: ScoreSummary | None) -> float | str:
    return "" if summary is None else round(summary.se, 4)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "task_id",
        "metric",
        "higher_better",
        "baseline",
        "scaffold",
        "baseline_mean",
        "baseline_se",
        "baseline_n",
        "scaffold_mean",
        "scaffold_se",
        "scaffold_n",
        "improvement",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Baseline vs Scaffold",
        "",
        "| task_id | metric | higher_better | baseline | scaffold | improvement |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            (
                "| {task_id} | {metric} | {higher_better} | "
                "{baseline} | {scaffold} | {improvement} |"
            ).format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate EpisodeResult JSON files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "runs" / "fake_smoke",
        help="Directory containing EpisodeResult JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "reports" / "fake_smoke_summary",
        help="Output path without extension; .csv and .md are written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
