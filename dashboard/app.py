import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
from dashboard.components.run_table import render_run_table
from dashboard.components.step_timeline import render_timeline
from dashboard.components.metrics_chart import render_metrics_charts

st.set_page_config(page_title="MLGym Coach Dashboard", layout="wide")
st.title("📊 MLGym Coach Dashboard")

# Загрузка данных
DATA_DIR = Path("reports")
example_data = Path("dashboard/example_data")

def load_episodes():
    episodes = []
    for json_file in list(DATA_DIR.glob("*.json")) + list(example_data.glob("*.json")):
        with open(json_file) as f:
            episodes.append(json.load(f))
    return episodes

episodes = load_episodes()

# Sidebar
st.sidebar.header("Фильтры")
agent_types = list(set(ep["agent_type"] for ep in episodes))
selected_agents = st.sidebar.multiselect("Тип агента", agent_types, default=agent_types)

filtered_episodes = [ep for ep in episodes if ep["agent_type"] in selected_agents]

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Список прогонов", "🔍 Детали прогона", "⚖️ Сравнение"])

with tab1:
    render_run_table(filtered_episodes)

with tab2:
    if filtered_episodes:
        selected_ep = st.selectbox(
            "Выберите прогон",
            options=[ep["episode_id"] for ep in filtered_episodes]
        )
        episode = next(ep for ep in filtered_episodes if ep["episode_id"] == selected_ep)
        render_timeline(episode)
        render_metrics_charts(episode)

with tab3:
    if len(filtered_episodes) >= 2:
        ep1_id, ep2_id = st.columns(2)
        with ep1_id:
            sel1 = st.selectbox("Первый прогон", [ep["episode_id"] for ep in filtered_episodes], key="cmp1")
        with ep2_id:
            sel2 = st.selectbox("Второй прогон", [ep["episode_id"] for ep in filtered_episodes], key="cmp2")
        
        if sel1 != sel2:
            ep1 = next(ep for ep in filtered_episodes if ep["episode_id"] == sel1)
            ep2 = next(ep for ep in filtered_episodes if ep["episode_id"] == sel2)
            st.write("### Сравнение метрик")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Score", f"{ep1['final_score']:.3f}")
                st.metric("Total Tokens", f"{ep1['total_tokens']:,}")
            with col2:
                st.metric("Final Score", f"{ep2['final_score']:.3f}")
                st.metric("Total Tokens", f"{ep2['total_tokens']:,}")