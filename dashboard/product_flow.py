"""Главный продуктовый поток: новая задача → запуск → живой просмотр → результат."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from core.types import EpisodeResult, Stage
from dashboard.backend_client import (
    METRIC_HIGHER_BETTER,
    RunLaunch,
    api_base_url,
    read_run_meta,
    run_status,
    submit_run,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO_ROOT / "runs"

STAGES_ORDER = [Stage.EDA, Stage.BASELINE, Stage.IMPROVE, Stage.SUBMIT]
STAGE_LABELS = {
    Stage.EDA: "EDA",
    Stage.BASELINE: "Baseline",
    Stage.IMPROVE: "Improve",
    Stage.SUBMIT: "Submit",
}
METRIC_OPTIONS = ["roc_auc", "accuracy", "f1", "rmse", "mae"]
HINT_COLORS = {1: ("#e0f2fe", "#0284c7"), 2: ("#fff7ed", "#ea580c"), 3: ("#fef2f2", "#dc2626")}


def _init_session() -> None:
    defaults = {
        "product_phase": "create",  # create | live | final
        "active_run_id": None,
        "active_task_id": None,
        "active_run_dir": None,
        "last_launch": None,
        "prefill_agent": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_partial_episode(run_dir: Path) -> Optional[EpisodeResult]:
    partial_path = run_dir / "episode.partial.json"
    if partial_path.exists():
        try:
            return EpisodeResult.from_json(partial_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    # Fallback: первый готовый seed_*.json (пока Ваня не пишет partial).
    seed_files = sorted(run_dir.rglob("seed_*.json"))
    if seed_files:
        try:
            return EpisodeResult.from_json(seed_files[0].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    return None


def load_final_episode(run_dir: Path) -> Optional[EpisodeResult]:
    final_path = run_dir / "episode.json"
    if final_path.exists():
        try:
            return EpisodeResult.from_json(final_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    seed_files = sorted(run_dir.rglob("seed_*.json"))
    if not seed_files:
        return None
    try:
        return EpisodeResult.from_json(seed_files[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def total_tokens_so_far(episode: Optional[EpisodeResult]) -> int:
    if episode is None:
        return 0
    if episode.total_tokens:
        return episode.total_tokens
    return sum(step.tokens_used for step in episode.steps)


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


def render_live_timeline(episode: EpisodeResult) -> None:
    if not episode.steps:
        st.info("Шаги появятся по мере выполнения агента…")
        return
    for step in episode.steps:
        title = f"Step {step.idx} | {step.stage.value} | {step.action.type.value}"
        with st.expander(title, expanded=(step.idx == episode.steps[-1].idx)):
            st.markdown(f"**Action:** `{step.action.type.value}`")
            st.code(step.action.content or "—", language="python")
            st.markdown(f"**Result:** {step.result}")
            c1, c2 = st.columns(2)
            c1.metric("Tokens (step)", step.tokens_used)
            c2.metric("Val score", f"{step.val_score:.3f}" if step.val_score is not None else "—")
            for hint in step.hints:
                bg, border = HINT_COLORS.get(hint.level, ("#f3f4f6", "#9ca3af"))
                st.markdown(
                    f"""
                    <div style="background:{bg};border:1px solid {border};padding:8px;border-radius:8px;">
                        <b>L{hint.level}</b> | {hint.stage.value} | {hint.item_id}<br>{hint.text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_create_form() -> None:
    st.subheader("1. Новая задача")
    backend = api_base_url() or "local fallback"
    st.caption(f"Бэкенд: {backend}")

    with st.form("new_task_form", clear_on_submit=False):
        description = st.text_area(
            "Описание задачи",
            placeholder="Бинарная классификация оттока: предсказать churn по табличным признакам…",
            height=120,
        )
        metric = st.selectbox("Метрика", METRIC_OPTIONS, index=0)

        col1, col2 = st.columns(2)
        train_csv = col1.file_uploader("train.csv (обязательно)", type=["csv"])
        sample_csv = col2.file_uploader("sample.csv (опционально)", type=["csv"])

        col3, col4 = st.columns(2)
        test_features_csv = col3.file_uploader("test_features.csv (обязательно)", type=["csv"])
        hidden_labels_csv = col4.file_uploader("hidden_labels.csv (опционально)", type=["csv"])

        c1, c2, c3 = st.columns(3)
        token_budget = c1.number_input("Токен-бюджет", min_value=500, max_value=100_000, value=8000, step=500)
        prefill = st.session_state.get("prefill_agent")
        agent_index = 1 if prefill == "scaffold" else 0
        agent = c2.selectbox("Агент", ["baseline", "scaffold"], index=agent_index)
        seeds = c3.number_input("Число сидов", min_value=1, max_value=20, value=1)

        llm_mode = st.selectbox("Режим LLM", ["mock", "real"], index=0)

        submitted = st.form_submit_button("Запустить", type="primary", use_container_width=True)

    if not submitted:
        return

    if not description.strip():
        st.error("Добавьте описание задачи.")
        return
    if train_csv is None or test_features_csv is None:
        st.error("Нужны train.csv и test_features.csv.")
        return

    with st.spinner("Отправка на бэкенд и запуск раннера…"):
        try:
            launch: RunLaunch = submit_run(
                description=description.strip(),
                metric=metric,
                agent=agent,
                token_budget=int(token_budget),
                seeds=int(seeds),
                train_csv=train_csv,
                test_features_csv=test_features_csv,
                sample_csv=sample_csv,
                hidden_labels_csv=hidden_labels_csv,
                llm_mode=llm_mode,
            )
        except Exception as exc:
            st.error(f"Не удалось запустить: {exc}")
            return

    st.session_state.product_phase = "live"
    st.session_state.active_run_id = launch.run_id
    st.session_state.active_task_id = launch.task_id
    st.session_state.active_run_dir = str(launch.run_dir)
    st.session_state.prefill_agent = None
    st.session_state.last_launch = {
        "task_id": launch.task_id,
        "run_id": launch.run_id,
        "agent": agent,
        "metric": metric,
        "token_budget": int(token_budget),
        "seeds": int(seeds),
        "backend": launch.backend,
    }
    st.success(f"Запуск создан: task_id={launch.task_id}, run_id={launch.run_id}")
    st.rerun()


def _render_live_view() -> None:
    run_id = st.session_state.active_run_id
    run_dir = Path(st.session_state.active_run_dir)
    meta = read_run_meta(run_dir)

    st.subheader("2. Живой прогон")
    st.caption(f"run_id: `{run_id}` · task_id: `{st.session_state.active_task_id}`")

    status = run_status(run_dir)
    partial = load_partial_episode(run_dir)
    tokens = total_tokens_so_far(partial)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Статус", status)
    c2.metric("Токены (бегущий счётчик)", f"{tokens:,}")
    c3.metric("Шагов", len(partial.steps) if partial else 0)
    c4.metric("Агент", meta.get("agent", "—"))

    render_progress_stepper(partial.steps if partial else [])

    if partial:
        render_live_timeline(partial)
    else:
        st.info("Ожидаем `runs/<run_id>/episode.partial.json` (пишет env)…")

    with st.expander("Лог раннера"):
        log_path = run_dir / "runner.log"
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8")[-8000:], language="text")
        else:
            st.write("Лог пока пуст.")

    if status == "completed":
        final = load_final_episode(run_dir)
        if final and final.final_test_score is not None:
            final_path = run_dir / "episode.json"
            if not final_path.exists():
                final_path.write_text(final.to_json(), encoding="utf-8")
        st.session_state.product_phase = "final"
        st.rerun()

    if status == "failed":
        st.error("Прогон завершился с ошибкой. Проверьте runner.log.")
        if st.button("Вернуться к форме"):
            st.session_state.product_phase = "create"
            st.rerun()
        return

    time.sleep(1.5)
    st.rerun()


def _render_final_view() -> None:
    run_dir = Path(st.session_state.active_run_dir)
    episode = load_final_episode(run_dir)
    launch = st.session_state.last_launch or {}

    st.subheader("3. Результат")
    if episode is None:
        st.warning("Финальный EpisodeResult ещё не найден.")
        if st.button("Вернуться к форме"):
            st.session_state.product_phase = "create"
            st.rerun()
        return

    c1, c2, c3 = st.columns(3)
    score = episode.final_test_score
    c1.metric("Final test score", f"{score:.4f}" if score is not None else "—")
    c2.metric("Checklist coverage", f"{episode.checklist_coverage:.0%}")
    c3.metric("Total tokens", f"{episode.total_tokens:,}")

    st.markdown(
        f"""
        - **task_id:** `{episode.task_id}`
        - **agent:** `{episode.agent}`
        - **seed:** `{episode.seed}`
        - **metric:** `{launch.get('metric', '—')}` ({'↑' if METRIC_HIGHER_BETTER.get(launch.get('metric', ''), True) else '↓'} лучше)
        """
    )

    render_progress_stepper(episode.steps)
    render_live_timeline(episode)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Запустить ещё раз", use_container_width=True):
            st.session_state.product_phase = "create"
            st.session_state.prefill_agent = episode.agent
            st.rerun()
    with col_b:
        other = "scaffold" if episode.agent == "baseline" else "baseline"
        if st.button(f"Сравнить с {other}", use_container_width=True):
            st.session_state.product_phase = "create"
            st.session_state.prefill_agent = other
            st.rerun()


def page_product_flow() -> None:
    _init_session()

    st.title("mlgym-coach")
    st.markdown("**Новая задача → запуск → живой просмотр → результат**")

    phase = st.session_state.product_phase
    if phase == "create":
        if st.session_state.prefill_agent:
            st.info(f"Подсказка: для сравнения выбран агент `{st.session_state.prefill_agent}`")
        _render_create_form()
    elif phase == "live":
        _render_live_view()
    elif phase == "final":
        _render_final_view()
    else:
        st.session_state.product_phase = "create"
        st.rerun()
