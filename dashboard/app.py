"""Streamlit-вьюер прогонов поверх EpisodeResult.json.

Запуск из корня репозитория:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.types import EpisodeResult, Stage

EXAMPLES_DIR = _REPO_ROOT / "examples"
REPORTS_DIR = _REPO_ROOT / "reports"

STAGES_ORDER = [
    Stage.UNDERSTAND,
    Stage.EDA,
    Stage.BASELINE,
    Stage.IMPROVE,
    Stage.SUBMIT,
]

STAGE_LABELS = {
    Stage.UNDERSTAND: "Understand",
    Stage.EDA: "EDA",
    Stage.BASELINE: "Baseline",
    Stage.IMPROVE: "Improve",
    Stage.SUBMIT: "Submit",
}

HINT_COLORS = {
    1: ("#e3f2fd", "#2196f3"),
    2: ("#fff3e0", "#ff9800"),
    3: ("#ffebee", "#f44336"),
}

st.set_page_config(page_title="MLGYM Dashboard", layout="wide")


@st.cache_data
def load_episodes() -> list[EpisodeResult]:
    """Загружает EpisodeResult из examples/ и reports/."""
    episodes: list[EpisodeResult] = []
    for directory in (EXAMPLES_DIR, REPORTS_DIR):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            episodes.append(EpisodeResult.from_json(path.read_text(encoding="utf-8")))
    return episodes


def episode_label(ep: EpisodeResult) -> str:
    score = ep.final_test_score
    score_str = f"{score:.3f}" if score is not None else "—"
    return f"{ep.task_id} · {ep.agent} · seed {ep.seed} · score {score_str}"


def render_progress_bar(steps: list) -> None:
    visited = {step.stage for step in steps}
    current = steps[-1].stage if steps else Stage.UNDERSTAND

    cols = st.columns(len(STAGES_ORDER))
    for i, stage in enumerate(STAGES_ORDER):
        if stage not in visited:
            status, color = "Locked", "gray"
        elif stage == current:
            status, color = "Current", "orange"
        else:
            status, color = "Done", "green"

        with cols[i]:
            st.markdown(
                f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;
                            text-align: center; border-left: 5px solid {color};'>
                    <small style='color: gray; text-transform: uppercase;'>Stage {i + 1}</small><br>
                    <b>{STAGE_LABELS[stage]}</b><br>
                    <span style='font-size: 0.8em'>{status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_hints(hints: list) -> None:
    for hint in hints:
        bg, border = HINT_COLORS.get(hint.level, ("#f5f5f5", "#ccc"))
        st.markdown(
            f"""
            <div style='background-color: {bg}; padding: 10px; border-radius: 5px;
                        border: 1px solid {border}; margin-top: 8px;'>
                <b>Hint L{hint.level}</b> · {hint.stage.value} / {hint.item_id}<br>
                {hint.text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_timeline(steps: list) -> None:
    for step in steps:
        title = f"Step {step.idx}: {STAGE_LABELS[step.stage]} · {step.action.type.value}"
        with st.expander(title):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Action ({step.action.type.value}):**")
                st.code(step.action.content or "—", language="python")
                st.markdown(f"**Result:** {step.result}")
            with col2:
                st.metric("Tokens", step.tokens_used)
                val = step.val_score
                st.metric("Val score", f"{val:.3f}" if val is not None else "—")
            if step.hints:
                render_hints(step.hints)


def render_charts(steps: list, title: str) -> None:
    if not steps:
        st.info("No steps to chart.")
        return

    df = pd.DataFrame(
        {
            "idx": [s.idx for s in steps],
            "tokens_used": [s.tokens_used for s in steps],
            "val_score": [s.val_score for s in steps],
        }
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tokens per step", "Val score per step"))
    fig.add_trace(
        go.Scatter(x=df["idx"], y=df["tokens_used"], mode="lines+markers", name="Tokens"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["idx"],
            y=df["val_score"],
            mode="lines+markers",
            name="Val score",
            line=dict(color="green"),
            connectgaps=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=300, showlegend=False, title=title)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("MLGYM-COACH Dashboard")

    episodes = load_episodes()
    if not episodes:
        st.warning("No episodes found in examples/ or reports/")
        return

    ep_options = {episode_label(ep): ep for ep in episodes}

    tab1, tab2, tab3 = st.tabs(["Список прогонов", "Детали прогона", "Сравнение"])

    with tab1:
        st.header("Список прогонов")
        rows = [
            {
                "Task": ep.task_id,
                "Agent": ep.agent,
                "Seed": ep.seed,
                "Final test score": ep.final_test_score,
                "Total tokens": ep.total_tokens,
                "Checklist coverage": ep.checklist_coverage,
            }
            for ep in episodes
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        scores = [ep.final_test_score for ep in episodes if ep.final_test_score is not None]
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего прогонов", len(episodes))
        if scores:
            c2.metric("Лучший test score", f"{max(scores):.3f}")
            c3.metric("Средний test score", f"{sum(scores) / len(scores):.3f}")

    with tab2:
        st.header("Детали прогона")
        selected_key = st.selectbox("Выберите прогон:", list(ep_options.keys()))
        ep = ep_options[selected_key]

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Agent", ep.agent)
        col_b.metric("Final test score", f"{ep.final_test_score:.3f}" if ep.final_test_score else "—")
        col_c.metric("Tokens", f"{ep.total_tokens:,}")
        col_d.metric("Checklist", f"{ep.checklist_coverage:.0%}")

        st.subheader("Прогресс по стадиям")
        render_progress_bar(ep.steps)

        st.subheader("Graphs")
        render_charts(ep.steps, episode_label(ep))

        st.subheader("Timeline")
        render_timeline(ep.steps)

    with tab3:
        st.header("Сравнение двух прогонов")
        if len(episodes) < 2:
            st.info("Нужно минимум 2 прогона для сравнения.")
            return

        keys = list(ep_options.keys())
        col1, col2 = st.columns(2)
        with col1:
            ep1_key = st.selectbox("Прогон A", keys, key="cmp_a")
        with col2:
            ep2_key = st.selectbox("Прогон B", keys, index=1, key="cmp_b")

        if ep1_key == ep2_key:
            st.error("Выберите разные прогоны.")
            return

        ep1 = ep_options[ep1_key]
        ep2 = ep_options[ep2_key]

        st.subheader("Метрики")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"### {ep1.agent}")
            st.metric("Test score", f"{ep1.final_test_score:.3f}" if ep1.final_test_score else "—")
            st.metric("Tokens", f"{ep1.total_tokens:,}")
            st.metric("Steps", len(ep1.steps))
            st.metric("Checklist", f"{ep1.checklist_coverage:.0%}")
        with col_m2:
            st.markdown(f"### {ep2.agent}")
            st.metric("Test score", f"{ep2.final_test_score:.3f}" if ep2.final_test_score else "—")
            st.metric("Tokens", f"{ep2.total_tokens:,}")
            st.metric("Steps", len(ep2.steps))
            st.metric("Checklist", f"{ep2.checklist_coverage:.0%}")

        st.subheader("Graphs side by side")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            render_charts(ep1.steps, ep1.agent)
        with col_g2:
            render_charts(ep2.steps, ep2.agent)


if __name__ == "__main__":
    main()
