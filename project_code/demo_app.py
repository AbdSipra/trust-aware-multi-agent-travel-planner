from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.eval.runner import run_experiment


st.title("Trust-Aware Multi-Agent Travel Planner Demo")
task_split = st.selectbox("Task split", ["dev_tasks", "clean_eval_tasks", "attacked_eval_tasks"])
system_variant = st.selectbox(
    "System variant",
    ["single_agent_tool_use", "naive_multi_agent_shared_memory", "trust_aware_multi_agent"],
)
attack_mode = st.text_input("Attack mode override", "")

if st.button("Run Experiment"):
    metrics, traces = run_experiment(
        task_split=task_split,
        system_variant=system_variant,
        attack_mode=attack_mode or None,
        task_limit=5,
    )
    st.subheader("Metrics")
    st.json(metrics)
    if traces:
        st.subheader("Latest Trace")
        st.json(traces[0].to_dict())
