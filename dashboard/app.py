"""Web dashboard for mlgym-coach runs.

Run from repository root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.types import EpisodeResult, Stage  # noqa: E402
from dashboard.product_flow import page_product_flow  # noqa: E402

EXAMPLES_DIR = _REPO_ROOT / "examples"
REPORTS_DIR = _REPO_ROOT / "reports"
TASKS_DIR = _REPO_ROOT / "tasks"
TASK_SPECS_DIR = TASKS_DIR / "specs"
RUNS_DIR = _REPO_ROOT / "runs"
RUNNER_CONFIGS_DIR = RUNS_DIR / "_dashboard_configs"

STAGES_ORDER = [
    Stage.EDA,
    Stage.BASELINE,
    Stage.IMPROVE,
    Stage.SUBMIT,
]

STAGE_LABELS = {
    Stage.EDA: "EDA",
    Stage.BASELINE: "Baseline",
    Stage.IMPROVE: "Improve",
    Stage.SUBMIT: "Submit",
}

AGENT_COLORS = {"baseline": "#4f46e5", "scaffold": "#10b981"}
HINT_COLORS = {1: ("#e0f2fe", "#0284c7"), 2: ("#fff7ed", "#ea580c"), 3: ("#fef2f2", "#dc2626")}

MOCK_TASK_CATALOG = {
    "churn_small": {
        "name": "Churn Small",
        "description": "Binary churn classification on tabular customer features.",
        "metric": "roc_auc",
    },
    "house_prices": {
        "name": "House Prices",
        "description": "Regression baseline for tabular housing data.",
        "metric": "rmse",
    },
}


st.set_page_config(page_title="mlgym-coach dashboard", page_icon=":bar_chart:", layout="wide")


@st.cache_data(ttl=5)
def load_episodes_with_meta() -> list[dict]:
    episodes: list[dict] = []
    for directory in (EXAMPLES_DIR, REPORTS_DIR):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            ep = EpisodeResult.from_json(path.read_text(encoding="utf-8"))
            episodes.append(
                {
                    "id": f"{ep.task_id}:{ep.agent}:seed{ep.seed}:{path.stem}",
                    "episode": ep,
                    "source": "examples" if directory == EXAMPLES_DIR else "reports",
                    "file": path.name,
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime),
                }
            )
    if RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.rglob("seed_*.json")):
            if "_dashboard_configs" in path.parts:
                continue
            ep = EpisodeResult.from_json(path.read_text(encoding="utf-8"))
            episodes.append(
                {
                    "id": f"{ep.task_id}:{ep.agent}:seed{ep.seed}:{path.stem}:{path.parent.name}",
                    "episode": ep,
                    "source": "runs",
                    "file": str(path.relative_to(_REPO_ROOT)),
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime),
                }
            )
    episodes.sort(key=lambda x: x["mtime"], reverse=True)
    return episodes


def load_tasks(episodes_with_meta: list[dict]) -> list[dict]:
    tasks: dict[str, dict] = {}

    # Start from on-disk task descriptors if they exist.
    if TASKS_DIR.exists():
        for path in list(TASKS_DIR.glob("*.json")) + list(TASKS_DIR.glob("*.yaml")) + list(TASKS_DIR.glob("*.yml")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {}
            except Exception:
                payload = {}
            task_id = payload.get("id") or path.stem
            tasks[task_id] = {
                "id": task_id,
                "name": payload.get("name", task_id),
                "description": payload.get("description", "No description yet."),
                "metric": payload.get("metric", "unknown"),
                "source": "tasks/",
            }

    # Enrich from known mock catalog + discovered episodes.
    for item in episodes_with_meta:
        task_id = item["episode"].task_id
        if task_id not in tasks:
            defaults = MOCK_TASK_CATALOG.get(task_id, {})
            tasks[task_id] = {
                "id": task_id,
                "name": defaults.get("name", task_id),
                "description": defaults.get("description", "Task discovered from EpisodeResult examples."),
                "metric": defaults.get("metric", "unknown"),
                "source": "episodes",
            }

    for task_id, defaults in MOCK_TASK_CATALOG.items():
        tasks.setdefault(
            task_id,
            {
                "id": task_id,
                "name": defaults["name"],
                "description": defaults["description"],
                "metric": defaults["metric"],
                "source": "mock",
            },
        )

    return sorted(tasks.values(), key=lambda x: x["id"])


def load_task_specs() -> dict[str, dict]:
    specs: dict[str, dict] = {}
    if not TASK_SPECS_DIR.exists():
        return specs

    for path in sorted(TASK_SPECS_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("id", path.stem))
        specs[task_id] = {
            "id": task_id,
            "description": str(raw.get("description", "No description yet.")),
            "metric": str(raw.get("metric", "unknown")),
            "spec_path": str(path.relative_to(_REPO_ROOT)),
        }
    return specs


def episode_label(meta: dict) -> str:
    ep = meta["episode"]
    score = f"{ep.final_test_score:.3f}" if ep.final_test_score is not None else "-"
    return f"{ep.task_id} | {ep.agent} | seed {ep.seed} | score {score} | {meta['file']}"


def episodes_dataframe(episodes_with_meta: list[dict]) -> pd.DataFrame:
    rows = []
    for item in episodes_with_meta:
        ep = item["episode"]
        rows.append(
            {
                "run_id": item["id"],
                "task_id": ep.task_id,
                "agent": ep.agent,
                "seed": ep.seed,
                "final_test_score": ep.final_test_score,
                "checklist_coverage": ep.checklist_coverage,
                "total_tokens": ep.total_tokens,
                "steps": len(ep.steps),
                "updated_at": item["mtime"],
                "source": item["source"],
                "file": item["file"],
            }
        )
    return pd.DataFrame(rows)


def render_cards(metrics: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(metrics))
    for idx, (title, value, subtitle) in enumerate(metrics):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="padding:14px;border:1px solid rgba(120,120,120,0.25);border-radius:12px;">
                    <div style="font-size:12px;opacity:0.8;">{title}</div>
                    <div style="font-size:30px;font-weight:700;line-height:1.1;">{value}</div>
                    <div style="font-size:12px;opacity:0.7;">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_progress_stepper(steps: list) -> None:
    visited = {s.stage for s in steps}
    current = steps[-1].stage if steps else Stage.EDA

    cols = st.columns(4)
    for idx, stage in enumerate(STAGES_ORDER):
        if stage not in visited:
            status, color, icon = "Locked", "#9ca3af", "-"
        elif stage == current:
            status, color, icon = "Current", "#f59e0b", "..."
        else:
            status, color, icon = "Done", "#10b981", "OK"

        with cols[idx]:
            st.markdown(
                f"""
                <div style="padding:12px;border:1px solid {color};border-radius:10px;text-align:center;">
                    <div style="font-size:11px;opacity:0.75;">Stage {idx + 1}</div>
                    <div style="font-weight:700;">{STAGE_LABELS[stage]}</div>
                    <div style="font-size:12px;color:{color};">{icon} {status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_hints(hints: list) -> None:
    for hint in hints:
        bg, border = HINT_COLORS.get(hint.level, ("#f3f4f6", "#9ca3af"))
        st.markdown(
            f"""
            <div style="background:{bg};border:1px solid {border};padding:10px;border-radius:10px;margin-top:6px;">
                <b>L{hint.level}</b> | {hint.stage.value} | {hint.item_id}<br>{hint.text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_timeline(steps: list) -> None:
    if not steps:
        st.info("No steps yet.")
        return

    for step in steps:
        title = f"Step {step.idx} | {step.stage.value} | {step.action.type.value}"
        with st.expander(title):
            left, right = st.columns([2, 1])
            with left:
                st.markdown("**Action content**")
                st.code(step.action.content or "-", language="python")
                st.markdown(f"**Environment result:** {step.result}")
            with right:
                st.metric("Tokens used", step.tokens_used)
                st.metric("Val score", f"{step.val_score:.3f}" if step.val_score is not None else "-")
            if step.hints:
                render_hints(step.hints)


def per_step_dataframe(episode: EpisodeResult) -> pd.DataFrame:
    cum_tokens = 0
    rows = []
    for step in episode.steps:
        cum_tokens += step.tokens_used
        rows.append(
            {
                "idx": step.idx,
                "val_score": step.val_score,
                "tokens_used": step.tokens_used,
                "tokens_cumulative": cum_tokens,
                "stage": step.stage.value,
            }
        )
    return pd.DataFrame(rows)


def render_step_charts(episode: EpisodeResult, title_suffix: str = "") -> None:
    df = per_step_dataframe(episode)
    if df.empty:
        st.info("No chart data.")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cumulative tokens", "Validation score by step"),
    )
    fig.add_trace(
        go.Scatter(x=df["idx"], y=df["tokens_cumulative"], mode="lines+markers", name="tokens", line=dict(color="#4f46e5")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["idx"], y=df["val_score"], mode="lines+markers", name="val_score", line=dict(color="#10b981")),
        row=1,
        col=2,
    )
    fig.update_layout(height=320, title=title_suffix, showlegend=False, margin=dict(l=10, r=10, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_overlay_chart(selected: list[dict], metric_key: str, title: str) -> None:
    fig = go.Figure()
    for item in selected:
        ep = item["episode"]
        df = per_step_dataframe(ep)
        if df.empty:
            continue
        y = df[metric_key]
        color = AGENT_COLORS.get(ep.agent, None)
        fig.add_trace(
            go.Scatter(
                x=df["idx"],
                y=y,
                mode="lines+markers",
                name=f"{ep.agent} seed={ep.seed}",
                line=dict(color=color) if color else None,
            )
        )
    fig.update_layout(height=320, title=title, margin=dict(l=10, r=10, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def mean_se(values: list[float]) -> str:
    if not values:
        return "-"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.3f} ± 0.000"
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    se = math.sqrt(variance / len(values))
    return f"{mean:.3f} ± {se:.3f}"


def page_overview(df: pd.DataFrame) -> None:
    st.title("mlgym-coach dashboard")
    st.caption("Единая точка просмотра задач, запусков и сравнения baseline vs scaffold")

    if df.empty:
        st.info("Еще нет прогонов. Добавьте JSON в examples/ или reports/.")
        return

    baseline = df[df["agent"] == "baseline"]
    scaffold = df[df["agent"] == "scaffold"]

    render_cards(
        [
            ("Total runs", str(len(df)), f"tasks: {df['task_id'].nunique()}"),
            ("Latest run", df.iloc[0]["file"], df.iloc[0]["updated_at"].strftime("%Y-%m-%d %H:%M")),
            (
                "Baseline mean score/tokens",
                f"{baseline['final_test_score'].mean():.3f}" if not baseline.empty else "-",
                f"tokens {int(baseline['total_tokens'].mean())}" if not baseline.empty else "-",
            ),
            (
                "Scaffold mean score/tokens",
                f"{scaffold['final_test_score'].mean():.3f}" if not scaffold.empty else "-",
                f"tokens {int(scaffold['total_tokens'].mean())}" if not scaffold.empty else "-",
            ),
        ]
    )

    st.subheader("Recent results")
    show_cols = ["task_id", "agent", "seed", "final_test_score", "total_tokens", "updated_at", "source", "file"]
    st.dataframe(df[show_cols].head(10), use_container_width=True, hide_index=True)


def page_tasks(tasks: list[dict], df: pd.DataFrame) -> None:
    st.title("Задачи")
    st.caption("Каталог задач из tasks/ и обнаруженных прогонов")

    if not tasks:
        st.info("Список задач пуст.")
        return

    for task in tasks:
        task_df = df[df["task_id"] == task["id"]] if not df.empty else pd.DataFrame()
        st.markdown(
            f"""
            <div style="padding:14px;border:1px solid rgba(120,120,120,0.25);border-radius:12px;margin-bottom:10px;">
                <div style="font-size:18px;font-weight:700;">{task['name']}</div>
                <div style="opacity:0.8;margin:6px 0;">{task['description']}</div>
                <div><b>id:</b> {task['id']} | <b>metric:</b> {task['metric']} | <b>source:</b> {task['source']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not task_df.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Runs", len(task_df))
            c2.metric("Best score", f"{task_df['final_test_score'].max():.3f}")
            c3.metric("Mean tokens", f"{int(task_df['total_tokens'].mean())}")


def page_run_experiment(tasks: list[dict]) -> None:
    st.title("Запуск эксперимента")
    st.caption("Запуск через runner.run в фоне; статус подхватывается по PID и runs/")

    if "launches" not in st.session_state:
        st.session_state.launches = []

    task_specs = load_task_specs()
    task_options = sorted(task_specs.keys())

    if not task_options:
        st.warning("Не найдены task specs в tasks/specs/*.yaml. Запуск недоступен.")
        return

    with st.form("launch_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        task_id = col1.selectbox("Задача", task_options)
        agent = col2.selectbox("Тип агента", ["baseline", "scaffold"])
        model = col3.text_input("Модель", value="fake-model")

        c4, c5, c6, c7 = st.columns(4)
        seeds = c4.number_input("Число сидов", min_value=1, max_value=20, value=3)
        budget = c5.number_input("Токен-бюджет", min_value=500, max_value=50000, value=8000, step=500)
        hint_level = c6.selectbox("Уровень подсказок", ["L1", "L2", "L3"])  # пока в config для runner metadata
        llm_mode = c7.selectbox("Режим LLM", ["mock", "real"], index=0)
        max_steps = st.number_input("Максимум шагов", min_value=2, max_value=50, value=6)

        submitted = st.form_submit_button("Запустить", use_container_width=True)
        if submitted:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = f"{task_id}-{agent}-{ts}-{uuid4().hex[:6]}"
            experiment_name = f"dashboard-{run_id}"

            config = {
                "experiment_name": experiment_name,
                "model": model,
                "env": "fake" if llm_mode == "mock" else "real",
                "agents": [agent],
                "seeds": list(range(int(seeds))),
                "tasks": [task_specs[task_id]["spec_path"]],
                "token_budget": int(budget),
                "max_steps": int(max_steps),
                "output_dir": "runs",
                # runner игнорирует неизвестные поля; оставляем для дебага/аудита.
                "hint_level": hint_level,
                "llm_mode": llm_mode,
            }

            RUNNER_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            cfg_rel = Path("runs") / "_dashboard_configs" / f"{experiment_name}.yaml"
            cfg_abs = _REPO_ROOT / cfg_rel
            cfg_abs.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

            output_dir = RUNS_DIR / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "runner.log"
            python_bin = _REPO_ROOT / ".venv" / "bin" / "python"
            cmd = [str(python_bin), "-m", "runner.run", "--config", str(cfg_abs)]
            env = os.environ.copy()
            env["MLGYM_LLM"] = llm_mode

            with log_path.open("a", encoding="utf-8") as log_file:
                proc = subprocess.Popen(  # noqa: S603
                    cmd,
                    cwd=str(_REPO_ROOT),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

            launch = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "experiment_name": experiment_name,
                "task_id": task_id,
                "agent": agent,
                "model": model,
                "seeds": int(seeds),
                "budget_tokens": int(budget),
                "hint_level": hint_level,
                "llm_mode": llm_mode,
                "config_path": str(cfg_rel),
                "output_dir": str(output_dir.relative_to(_REPO_ROOT)),
                "log_path": str(log_path.relative_to(_REPO_ROOT)),
                "pid": proc.pid,
                "status": "running",
            }
            st.session_state.launches.insert(0, launch)
            st.success(f"Эксперимент запущен в фоне (PID {proc.pid}).")

    if st.session_state.launches:
        for launch in st.session_state.launches:
            pid = launch.get("pid")
            output_dir = _REPO_ROOT / launch["output_dir"]
            run_files = list(output_dir.rglob("seed_*.json")) if output_dir.exists() else []
            launch["result_files"] = len(run_files)

            if launch["status"] in {"completed", "failed"}:
                continue

            running = False
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    running = True
                except OSError:
                    running = False

            if running:
                launch["status"] = "running"
            else:
                launch["status"] = "completed" if run_files else "failed"

        st.subheader("История запусков")
        st.dataframe(pd.DataFrame(st.session_state.launches), use_container_width=True, hide_index=True)
        st.button("Обновить статусы", use_container_width=True)


def page_runs_list(df: pd.DataFrame) -> None:
    st.title("Список прогонов")
    if df.empty:
        st.info("Еще нет прогонов.")
        return

    col1, col2, col3 = st.columns(3)
    task_filter = col1.selectbox("Фильтр по задаче", ["all"] + sorted(df["task_id"].unique().tolist()))
    agent_filter = col2.selectbox("Фильтр по агенту", ["all"] + sorted(df["agent"].unique().tolist()))
    sort_by = col3.selectbox("Сортировка", ["updated_at", "final_test_score", "total_tokens"])

    filtered = df.copy()
    if task_filter != "all":
        filtered = filtered[filtered["task_id"] == task_filter]
    if agent_filter != "all":
        filtered = filtered[filtered["agent"] == agent_filter]

    ascending = sort_by == "total_tokens"
    filtered = filtered.sort_values(sort_by, ascending=ascending)
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def page_run_details(episodes_with_meta: list[dict]) -> None:
    st.title("Один прогон")
    if not episodes_with_meta:
        st.info("Еще нет прогонов.")
        return

    options = {episode_label(item): item for item in episodes_with_meta}
    selected = st.selectbox("Выберите прогон", list(options.keys()))
    item = options[selected]
    ep = item["episode"]

    render_cards(
        [
            ("Task", ep.task_id, f"source: {item['source']} / {item['file']}"),
            ("Agent", ep.agent, f"seed {ep.seed}"),
            ("Final test score", f"{ep.final_test_score:.3f}" if ep.final_test_score is not None else "-", "higher is better"),
            ("Checklist coverage", f"{ep.checklist_coverage:.0%}", f"total tokens {ep.total_tokens}"),
        ]
    )

    st.subheader("Прогресс по стадиям")
    render_progress_stepper(ep.steps)

    st.subheader("Графики по шагам")
    render_step_charts(ep, title_suffix=f"{ep.task_id} | {ep.agent} | seed {ep.seed}")

    st.subheader("Таймлайн")
    render_timeline(ep.steps)


def page_compare(episodes_with_meta: list[dict]) -> None:
    st.title("Сравнение прогонов")
    if len(episodes_with_meta) < 2:
        st.info("Нужно минимум 2 прогона.")
        return

    labels = [episode_label(item) for item in episodes_with_meta]
    selected_labels = st.multiselect("Выберите 2+ прогона", labels, default=labels[:2])
    if len(selected_labels) < 2:
        st.warning("Выберите минимум два прогона для сравнения.")
        return

    selected = [episodes_with_meta[labels.index(label)] for label in selected_labels]

    left, right = st.columns(2)
    with left:
        render_overlay_chart(selected, "tokens_cumulative", "Cumulative tokens (overlay)")
    with right:
        render_overlay_chart(selected, "val_score", "Validation score (overlay)")

    rows = []
    for item in selected:
        ep = item["episode"]
        rows.append(
            {
                "task_id": ep.task_id,
                "agent": ep.agent,
                "seed": ep.seed,
                "final_test_score": ep.final_test_score,
                "checklist_coverage": ep.checklist_coverage,
                "total_tokens": ep.total_tokens,
                "steps": len(ep.steps),
            }
        )
    metrics_df = pd.DataFrame(rows)
    st.subheader("Метрики выбранных прогонов")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Агрегаты по группам (mean ± SE)")
    agg_rows = []
    for agent, group in metrics_df.groupby("agent"):
        agg_rows.append(
            {
                "agent": agent,
                "runs": len(group),
                "final_test_score": mean_se([x for x in group["final_test_score"].tolist() if pd.notna(x)]),
                "checklist_coverage": mean_se([x for x in group["checklist_coverage"].tolist() if pd.notna(x)]),
                "total_tokens": mean_se([float(x) for x in group["total_tokens"].tolist() if pd.notna(x)]),
            }
        )
    st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)


def apply_theme(dark_mode: bool) -> None:
    """Светлая тема — базовая (см. .streamlit/config.toml), её не трогаем.

    Для тёмного режима перекрашиваем фон, текст, ПОДПИСИ виджетов и поля ввода
    целиком — иначе светлые лейблы базовой темы становятся нечитаемыми на тёмном
    фоне (и наоборот). Это и был баг: фон красился, а лейблы — нет."""
    if not dark_mode:
        return

    bg = "#0b1220"
    text = "#e5e7eb"
    panel = "#111827"
    field_bg = "#1f2937"
    border = "#374151"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        section[data-testid="stSidebar"] {{ background: {panel}; }}
        /* Заголовки, абзацы и ПОДПИСИ всех виджетов */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp span,
        .stApp label, .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stMarkdownContainer"] {{ color: {text}; }}
        /* Поля ввода: text/number/textarea/select */
        .stApp [data-baseweb="input"] input,
        .stApp [data-baseweb="textarea"] textarea,
        .stApp [data-baseweb="select"] div {{
            color: {text}; background: {field_bg};
        }}
        .stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"],
        .stApp [data-baseweb="select"] > div {{
            background: {field_bg}; border-color: {border};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    episodes_with_meta = load_episodes_with_meta()
    df = episodes_dataframe(episodes_with_meta) if episodes_with_meta else pd.DataFrame()
    tasks = load_tasks(episodes_with_meta)

    with st.sidebar:
        st.title("mlgym-coach")
        dark_mode = st.toggle("Dark theme", value=False)
        page = st.radio(
            "Навигация",
            [
                "New task",
                "Overview",
                "Tasks",
                "Run experiment",
                "Runs list",
                "Run details",
                "Compare",
            ],
        )
        st.caption(f"Loaded runs: {len(episodes_with_meta)}")

    apply_theme(dark_mode)

    if page == "New task":
        page_product_flow()
    elif page == "Overview":
        page_overview(df)
    elif page == "Tasks":
        page_tasks(tasks, df)
    elif page == "Run experiment":
        page_run_experiment(tasks)
    elif page == "Runs list":
        page_runs_list(df)
    elif page == "Run details":
        page_run_details(episodes_with_meta)
    elif page == "Compare":
        page_compare(episodes_with_meta)


if __name__ == "__main__":
    main()
