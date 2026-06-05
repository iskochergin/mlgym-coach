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
from dashboard.pricing import DEFAULT_MODEL, dollars_to_tokens, tokens_to_dollars
from dashboard.i18n import tr

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO_ROOT / "runs"

STAGES_ORDER = [Stage.EDA, Stage.BASELINE, Stage.IMPROVE, Stage.SUBMIT]
STAGE_LABELS = {
    Stage.EDA: "EDA",
    Stage.BASELINE: "Baseline",
    Stage.IMPROVE: "Improve",
    Stage.SUBMIT: "Submit",
}
METRIC_OPTIONS = ["roc_auc", "rmse"]
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

    # Fallback: первый готовый episode.json (пока Ваня не пишет partial).
    seed_files = sorted(run_dir.rglob("episode.json"))
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

    seed_files = sorted(run_dir.rglob("episode.json"))
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
    lang = st.session_state.get("lang", "ru")
    visited = {s.stage for s in steps}
    current = steps[-1].stage if steps else Stage.EDA
    cols = st.columns(4)
    for idx, stage in enumerate(STAGES_ORDER):
        if stage not in visited:
            status, color, icon = tr(lang, "step.locked"), "#9ca3af", "-"
        elif stage == current:
            status, color, icon = tr(lang, "step.current"), "#f59e0b", "..."
        else:
            status, color, icon = tr(lang, "step.done"), "#10b981", "OK"
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
    lang = st.session_state.get("lang", "ru")
    if not episode.steps:
        st.info(tr(lang, "timeline.no_steps"))
        return
    for step in episode.steps:
        title = f"Step {step.idx} | {step.stage.value} | {step.action.type.value}"
        with st.expander(title, expanded=(step.idx == episode.steps[-1].idx)):
            st.markdown(f"**{tr(lang, 'timeline.action')}:** `{step.action.type.value}`")
            st.code(step.action.content or "—", language="python")
            st.markdown(f"**{tr(lang, 'timeline.result')}:** {step.result}")
            c1, c2 = st.columns(2)
            c1.metric(tr(lang, "timeline.tokens"), step.tokens_used)
            c2.metric(tr(lang, "timeline.val_score"), f"{step.val_score:.3f}" if step.val_score is not None else "—")
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


def _render_run_controls() -> tuple[int, str, int]:
    """Агент, число сидов и бюджет (токены ИЛИ $) с живым пересчётом.

    Рендерится вне st.form. Возвращает (token_budget:int, agent:str, seeds:int).
    На бэкенд/раннер уходит ровно целочисленный token_budget — конвертация
    только на UI, протокол не меняем."""
    lang = st.session_state.get("lang", "ru")
    st.markdown(
        "**Параметры запуска**" if lang == "ru" else ("**Run settings**" if lang == "en" else "**Մեկնարկի կարգավորումներ**")
    )

    prefill = st.session_state.get("prefill_agent")
    agent_index = 1 if prefill == "scaffold" else 0
    cA, cB = st.columns(2)
    agent = cA.selectbox(
        "Тип агента" if lang == "ru" else ("Agent" if lang == "en" else "Ագենտ"),
        ["baseline", "scaffold"],
        index=agent_index,
        key="run_agent",
    )
    seeds = int(
        cB.number_input(
            "Число сидов" if lang == "ru" else ("Seeds count" if lang == "en" else "Seed-երի քանակ"),
            min_value=1,
            max_value=20,
            value=1,
            key="run_seeds",
        )
    )

    mode = st.radio(
        "Единица бюджета" if lang == "ru" else ("Budget unit" if lang == "en" else "Բյուջեի միավոր"),
        ["Бюджет в токенах", "Бюджет в $"] if lang == "ru" else (
            ["Budget in tokens", "Budget in $"] if lang == "en" else ["Բյուջե token-ներով", "Բյուջե $-ով"]
        ),
        horizontal=True,
        key="budget_mode",
    )
    if mode in {"Budget in $", "Бюджет в $", "Բյուջե $-ով"}:
        dollars = st.number_input(
            "Бюджет, $" if lang == "ru" else ("Budget, $" if lang == "en" else "Բյուջե, $"),
            min_value=0.10,
            value=1.00,
            step=0.10,
            key="budget_dollars",
        )
        token_budget = dollars_to_tokens(dollars, DEFAULT_MODEL)
        st.caption(
            f"≈ {token_budget:,} токенов ({DEFAULT_MODEL})"
            if lang == "ru"
            else f"≈ {token_budget:,} tokens ({DEFAULT_MODEL})"
        )
    else:
        token_budget = int(
            st.number_input(
                "Токен-бюджет" if lang == "ru" else ("Token budget" if lang == "en" else "Տոկենների բյուջե"),
                min_value=500,
                max_value=100_000,
                value=8000,
                step=500,
                key="budget_tokens",
            )
        )
        st.caption(
            f"≈ ${tokens_to_dollars(token_budget, DEFAULT_MODEL):.2f} ({DEFAULT_MODEL})"
        )

    # Сводка на весь эксперимент. Эта форма запускает одного агента → множитель 1
    # (если появится мультивыбор baseline+scaffold, поставить agents_count=2).
    agents_count = 1
    total = tokens_to_dollars(token_budget * seeds * agents_count, DEFAULT_MODEL)
    agents_note = (
        (f" × {agents_count} агентов" if lang == "ru" else (f" × {agents_count} agents" if lang == "en" else f" × {agents_count} ագենտ"))
        if agents_count > 1
        else ""
    )
    if lang == "ru":
        st.caption(f"За эксперимент: {seeds} сидов × {token_budget:,} токенов{agents_note} ≈ ${total:.2f}")
    elif lang == "en":
        st.caption(f"Experiment total: {seeds} seeds × {token_budget:,} tokens{agents_note} ≈ ${total:.2f}")
    else:
        st.caption(f"Փորձարկման ընդհանուր՝ {seeds} seed × {token_budget:,} tokens{agents_note} ≈ ${total:.2f}")

    return token_budget, agent, seeds


def _render_create_form() -> None:
    lang = st.session_state.get("lang", "ru")
    st.subheader(f"1. {tr(lang, 'page.new_task')}")
    backend = api_base_url() or "local fallback"
    st.caption(f"Backend: {backend}")

    # Параметры запуска и бюджет — ВНЕ st.form: виджеты внутри формы не делают
    # rerun до сабмита, а нам нужен живой пересчёт бюджета токены⇄$.
    token_budget, agent, seeds = _render_run_controls()

    with st.form("new_task_form", clear_on_submit=False):
        description = st.text_area(
            "Описание задачи" if lang == "ru" else ("Task description" if lang == "en" else "Առաջադրանքի նկարագրություն"),
            placeholder="Бинарная классификация оттока: предсказать churn по табличным признакам…",
            height=120,
        )
        metric = st.selectbox("Метрика" if lang == "ru" else ("Metric" if lang == "en" else "Մետրիկա"), METRIC_OPTIONS, index=0)

        col1, col2 = st.columns(2)
        train_csv = col1.file_uploader(
            "train.csv (обязательно)" if lang == "ru" else ("train.csv (required)" if lang == "en" else "train.csv (պարտադիր)"),
            type=["csv"],
        )
        sample_csv = col2.file_uploader(
            "sample.csv (опционально)" if lang == "ru" else ("sample.csv (optional)" if lang == "en" else "sample.csv (ըստ ցանկության)"),
            type=["csv"],
        )

        col3, col4 = st.columns(2)
        test_features_csv = col3.file_uploader(
            "test_features.csv (обязательно)" if lang == "ru" else ("test_features.csv (required)" if lang == "en" else "test_features.csv (պարտադիր)"),
            type=["csv"],
        )
        hidden_labels_csv = col4.file_uploader(
            "hidden_labels.csv (опционально)" if lang == "ru" else ("hidden_labels.csv (optional)" if lang == "en" else "hidden_labels.csv (ըստ ցանկության)"),
            type=["csv"],
        )

        llm_mode = st.selectbox(
            "Режим LLM" if lang == "ru" else ("LLM mode" if lang == "en" else "LLM ռեժիմ"),
            ["mock", "ChatGPT", "DeepSeek"],
            index=0
        )

        submitted = st.form_submit_button(tr(lang, "runexp.submit"), type="primary", use_container_width=True)

    if not submitted:
        return

    if not description.strip():
        st.error(
            "Добавьте описание задачи."
            if lang == "ru"
            else ("Add task description." if lang == "en" else "Ավելացրեք առաջադրանքի նկարագրություն։")
        )
        return
    if train_csv is None or test_features_csv is None:
        st.error(
            "Нужны train.csv и test_features.csv."
            if lang == "ru"
            else ("train.csv and test_features.csv are required." if lang == "en" else "Պահանջվում են train.csv և test_features.csv։")
        )
        return

    with st.spinner(
        "Отправка на бэкенд и запуск раннера..."
        if lang == "ru"
        else ("Sending to backend and starting runner..." if lang == "en" else "Ուղարկում backend և runner-ի մեկնարկ...")
    ):
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
            st.error(
                f"Не удалось запустить: {exc}"
                if lang == "ru"
                else (f"Failed to launch: {exc}" if lang == "en" else f"Չհաջողվեց մեկնարկել՝ {exc}")
            )
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
    st.success(
        f"Запуск создан: task_id={launch.task_id}, run_id={launch.run_id}"
        if lang == "ru"
        else (f"Run created: task_id={launch.task_id}, run_id={launch.run_id}" if lang == "en" else f"Վազքը ստեղծվեց․ task_id={launch.task_id}, run_id={launch.run_id}")
    )
    st.rerun()


def _render_live_view() -> None:
    lang = st.session_state.get("lang", "ru")
    run_id = st.session_state.active_run_id
    run_dir = Path(st.session_state.active_run_dir)
    meta = read_run_meta(run_dir)

    st.subheader("2. Живой прогон" if lang == "ru" else ("2. Live run" if lang == "en" else "2. Կենդանի վազք"))
    st.caption(f"run_id: `{run_id}` · task_id: `{st.session_state.active_task_id}`")

    status = run_status(run_dir)
    partial = load_partial_episode(run_dir)
    tokens = total_tokens_so_far(partial)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Статус" if lang == "ru" else ("Status" if lang == "en" else "Կարգավիճակ"), status)
    c2.metric("Токены (бегущий счётчик)" if lang == "ru" else ("Tokens (running)" if lang == "en" else "Tokens (ընթացիկ)"), f"{tokens:,}")
    c3.metric("Шагов" if lang == "ru" else ("Steps" if lang == "en" else "Քայլեր"), len(partial.steps) if partial else 0)
    c4.metric("Агент" if lang == "ru" else ("Agent" if lang == "en" else "Ագենտ"), meta.get("agent", "—"))

    render_progress_stepper(partial.steps if partial else [])

    if partial:
        render_live_timeline(partial)
    else:
        st.info(
            "Ожидаем `runs/<run_id>/episode.partial.json`..."
            if lang == "ru"
            else ("Waiting for `runs/<run_id>/episode.partial.json`..." if lang == "en" else "Սպասում ենք `runs/<run_id>/episode.partial.json` ֆայլին...")
        )

    with st.expander("Лог раннера" if lang == "ru" else ("Runner log" if lang == "en" else "Runner-ի log")):
        log_path = run_dir / "runner.log"
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8")[-8000:], language="text")
        else:
            st.write("Лог пока пуст." if lang == "ru" else ("Log is empty yet." if lang == "en" else "Log-ը դեռ դատարկ է։"))

    if status == "completed":
        final = load_final_episode(run_dir)
        if final and final.final_test_score is not None:
            final_path = run_dir / "episode.json"
            if not final_path.exists():
                final_path.write_text(final.to_json(), encoding="utf-8")
        st.session_state.product_phase = "final"
        st.rerun()

    if status == "failed":
        st.error(
            "Прогон завершился с ошибкой. Проверьте runner.log."
            if lang == "ru"
            else ("Run failed. Check runner.log." if lang == "en" else "Վազքը ձախողվեց։ Ստուգեք runner.log-ը։")
        )
        if st.button("Вернуться к форме" if lang == "ru" else ("Back to form" if lang == "en" else "Վերադառնալ ձևին")):
            st.session_state.product_phase = "create"
            st.rerun()
        return

    time.sleep(1.5)
    st.rerun()


def _render_final_view() -> None:
    lang = st.session_state.get("lang", "ru")
    run_dir = Path(st.session_state.active_run_dir)
    episode = load_final_episode(run_dir)
    launch = st.session_state.last_launch or {}

    st.subheader("3. Результат" if lang == "ru" else ("3. Result" if lang == "en" else "3. Արդյունք"))
    if episode is None:
        st.warning(
            "Финальный EpisodeResult ещё не найден."
            if lang == "ru"
            else ("Final EpisodeResult is not found yet." if lang == "en" else "Վերջնական EpisodeResult-ը դեռ չի գտնվել։")
        )
        if st.button("Вернуться к форме" if lang == "ru" else ("Back to form" if lang == "en" else "Վերադառնալ ձևին")):
            st.session_state.product_phase = "create"
            st.rerun()
        return

    c1, c2, c3 = st.columns(3)
    score = episode.final_test_score
    c1.metric("Final test score" if lang == "en" else ("Финальный score" if lang == "ru" else "Վերջնական test score"), f"{score:.4f}" if score is not None else "—")
    c2.metric("Checklist coverage" if lang == "en" else ("Покрытие checklist" if lang == "ru" else "Checklist-ի ծածկույթ"), f"{episode.checklist_coverage:.0%}")
    c3.metric("Total tokens" if lang == "en" else ("Всего токенов" if lang == "ru" else "Ընդհանուր tokens"), f"{episode.total_tokens:,}")

    st.markdown(
        f"""
        - **task_id:** `{episode.task_id}`
        - **agent:** `{episode.agent}`
        - **seed:** `{episode.seed}`
        - **metric:** `{launch.get('metric', '—')}` ({
            ('выше лучше' if METRIC_HIGHER_BETTER.get(launch.get('metric', ''), True) else 'ниже лучше')
            if lang == 'ru'
            else (('higher is better' if METRIC_HIGHER_BETTER.get(launch.get('metric', ''), True) else 'lower is better')
                  if lang == 'en'
                  else ('բարձրն է լավ' if METRIC_HIGHER_BETTER.get(launch.get('metric', ''), True) else 'ցածրն է լավ'))
        })
        """
    )

    render_progress_stepper(episode.steps)
    render_live_timeline(episode)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Запустить ещё раз" if lang == "ru" else ("Run again" if lang == "en" else "Մեկնարկել կրկին"), use_container_width=True):
            st.session_state.product_phase = "create"
            st.session_state.prefill_agent = episode.agent
            st.rerun()
    with col_b:
        other = "scaffold" if episode.agent == "baseline" else "baseline"
        if st.button(
            (f"Сравнить с {other}" if lang == "ru" else (f"Compare with {other}" if lang == "en" else f"Համեմատել {other}-ի հետ")),
            use_container_width=True,
        ):
            st.session_state.product_phase = "create"
            st.session_state.prefill_agent = other
            st.rerun()


def page_product_flow() -> None:
    _init_session()
    lang = st.session_state.get("lang", "ru")

    st.title("mlgym-coach")
    st.markdown(f"**{tr(lang, 'flow.title')}**")

    phase = st.session_state.product_phase
    if phase == "create":
        if st.session_state.prefill_agent:
            if lang == "ru":
                st.info(f"Подсказка: для сравнения выбран агент `{st.session_state.prefill_agent}`")
            else:
                st.info(tr(lang, "flow.tip_compare", agent=st.session_state.prefill_agent))
        _render_create_form()
    elif phase == "live":
        _render_live_view()
    elif phase == "final":
        _render_final_view()
    else:
        st.session_state.product_phase = "create"
        st.rerun()
