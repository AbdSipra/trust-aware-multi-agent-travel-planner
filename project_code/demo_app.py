from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.demo.app import (
    available_attack_modes,
    canonical_summary_dir,
    extract_itinerary_frames,
    load_figure_paths,
    load_summary_tables,
    read_text_if_exists,
    run_demo,
    runtime_badges,
)


st.set_page_config(
    page_title="Trust-Aware Travel Planner",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(217, 164, 65, 0.16), transparent 22%),
        radial-gradient(circle at top left, rgba(31, 122, 140, 0.15), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%);
}

.hero {
    border-radius: 24px;
    padding: 1.4rem 1.6rem;
    background: linear-gradient(135deg, #16324f 0%, #1f7a8c 58%, #d9a441 100%);
    color: #ffffff;
    box-shadow: 0 18px 48px rgba(22, 50, 79, 0.18);
    margin-bottom: 1rem;
}

.hero h1 {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 2.2rem;
    margin: 0 0 0.4rem 0;
}

.hero p {
    font-size: 1rem;
    margin: 0.15rem 0;
    max-width: 60rem;
}

.status-strip {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin: 0.9rem 0 0.1rem 0;
}

.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    font-size: 0.9rem;
    font-weight: 600;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.28);
}

.panel {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(215, 221, 229, 0.95);
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 28px rgba(30, 37, 48, 0.05);
}

.section-title {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #1e2530;
    margin-bottom: 0.2rem;
}

.section-note {
    color: #6d7885;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
}

.figure-frame {
    background: #ffffff;
    border: 1px solid #d7dde5;
    border-radius: 18px;
    padding: 0.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 18px rgba(30, 37, 48, 0.04);
}

.small-note {
    color: #6d7885;
    font-size: 0.9rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_tables():
    return load_summary_tables(canonical_summary_dir())


@st.cache_data(show_spinner=False)
def _cached_attack_modes():
    return available_attack_modes()


@st.cache_data(show_spinner=False)
def _cached_figure_paths():
    return load_figure_paths(canonical_summary_dir())


@st.cache_data(show_spinner=False)
def _cached_notes():
    summary_dir = canonical_summary_dir()
    return {
        "figure_notes": read_text_if_exists(summary_dir / "figures" / "figure_notes.md"),
        "table_notes": read_text_if_exists(summary_dir / "tables" / "table_notes.md"),
    }


def _render_svg(path: Path, caption: str | None = None) -> None:
    svg = path.read_text(encoding="utf-8")
    st.markdown(f'<div class="figure-frame">{svg}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def _metric_row(metrics: dict) -> None:
    keys = [
        ("task_success", "Task Success"),
        ("contamination_spread", "Contamination"),
        ("recovery_rate", "Recovery"),
        ("attack_success", "Attack Success"),
        ("defensive_intervention", "Defensive Intervention"),
        ("latency_ms", "Latency (ms)"),
    ]
    cols = st.columns(len(keys))
    for col, (key, label) in zip(cols, keys, strict=False):
        value = metrics.get(key)
        if value is None:
            display = "--"
        elif isinstance(value, (int, float)) and key != "latency_ms":
            display = f"{float(value) * 100:.1f}%"
        elif isinstance(value, (int, float)):
            display = f"{float(value):.1f}"
        else:
            display = str(value)
        col.metric(label, display)


def _show_trace(trace: dict) -> None:
    itinerary_frames = extract_itinerary_frames(trace)

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown('<div class="section-title">Itinerary Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Structured selections chosen by the system for the sampled run.</div>', unsafe_allow_html=True)
        if not itinerary_frames:
            st.info("No finalized itinerary was produced for this trace.")
        else:
            for section_name, frame in itinerary_frames.items():
                st.markdown(f"**{section_name.title()}**")
                st.dataframe(frame, use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="section-title">Trust and Verification Trail</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Agent messages, quarantine events, and verifier outputs for the sampled run.</div>', unsafe_allow_html=True)
        if trace.get("quarantine_events"):
            st.warning(f"Quarantine events: {len(trace['quarantine_events'])}")
            st.json(trace["quarantine_events"])
        else:
            st.success("No quarantine events in this sampled run.")

        with st.expander("Verifier Decisions", expanded=True):
            st.json(trace.get("verifier_decisions", []))
        with st.expander("Agent Messages", expanded=False):
            st.json(trace.get("agent_messages", []))
        with st.expander("Raw Trace", expanded=False):
            st.json(trace)


badges = runtime_badges()
st.markdown(
    f"""
    <div class="hero">
        <h1>Trust-Aware Multi-Agent Travel Planner</h1>
        <p>
            Publication-focused demo for the constrained travel-planning benchmark.
            This view combines live experiment runs with the frozen Groq benchmark tables and figures.
        </p>
        <div class="status-strip">
            <span class="badge">Provider: {badges['provider']}</span>
            <span class="badge">Model: {badges['model']}</span>
            <span class="badge">LangGraph Available: {"Yes" if badges['langgraph_available'] else "No"}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Demo Controls")
    task_split = st.selectbox("Task split", ["dev_tasks", "clean_eval_tasks", "attacked_eval_tasks"])
    system_variant = st.selectbox(
        "System variant",
        [
            "single_agent_tool_use",
            "naive_multi_agent_shared_memory",
            "trust_aware_multi_agent",
            "ablation_no_provenance",
            "ablation_no_quarantine",
            "ablation_no_verifier",
        ],
    )
    attack_mode = st.selectbox("Attack override", [""] + _cached_attack_modes())
    task_limit = st.slider("Task limit", min_value=1, max_value=10, value=5)
    use_langgraph = st.checkbox(
        "Use LangGraph runtime",
        value=False,
        disabled=not bool(badges["langgraph_available"]),
        help="Optional execution path. The benchmark itself does not depend on LangGraph.",
    )
    persist_traces = st.checkbox("Persist traces to runs/", value=True)
    run_clicked = st.button("Run Experiment", use_container_width=True)
    st.markdown("---")
    st.caption(f"Canonical summary: `{badges['summary_dir']}`")

if run_clicked:
    with st.spinner("Running experiment and collecting trace data..."):
        result = run_demo(
            task_split=task_split,
            system_variant=system_variant,
            attack_mode=attack_mode or None,
            task_limit=task_limit,
            use_langgraph=use_langgraph,
            persist_traces=persist_traces,
        )
        st.session_state["demo_result"] = result
        st.session_state["demo_meta"] = {
            "task_split": task_split,
            "system_variant": system_variant,
            "attack_mode": attack_mode or "none",
            "task_limit": task_limit,
            "use_langgraph": use_langgraph,
        }

summary_frames = _cached_tables()
notes = _cached_notes()
figure_paths = _cached_figure_paths()
demo_result = st.session_state.get("demo_result")
demo_meta = st.session_state.get("demo_meta", {})

tab_run, tab_results, tab_figures, tab_notes = st.tabs(
    ["Run Lab", "Benchmark Tables", "Figures", "Research Notes"]
)

with tab_run:
    st.markdown('<div class="section-title">Live Experiment Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Run a small slice of the benchmark and inspect the trust-aware behavior step by step.</div>',
        unsafe_allow_html=True,
    )

    if demo_result:
        st.markdown(
            f'<div class="small-note">Latest run: split={demo_meta.get("task_split")} | variant={demo_meta.get("system_variant")} | attack={demo_meta.get("attack_mode")} | LangGraph={"on" if demo_meta.get("use_langgraph") else "off"}</div>',
            unsafe_allow_html=True,
        )
        _metric_row(demo_result["metrics"])
        st.markdown("---")
        _show_trace(demo_result["sample_trace"])
    else:
        st.info("Run an experiment from the sidebar to populate this lab view.")

with tab_results:
    st.markdown('<div class="section-title">Frozen Benchmark Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">These are the canonical full Groq summary artifacts used for the paper.</div>',
        unsafe_allow_html=True,
    )

    if "main_table" in summary_frames:
        st.markdown("**Main comparison table**")
        st.dataframe(summary_frames["main_table"], use_container_width=True, hide_index=True)
    if "attack_success_table" in summary_frames:
        st.markdown("**Attack-mode success table**")
        st.dataframe(summary_frames["attack_success_table"], use_container_width=True, hide_index=True)
    if "contamination_table" in summary_frames:
        st.markdown("**Attack-mode contamination table**")
        st.dataframe(summary_frames["contamination_table"], use_container_width=True, hide_index=True)

with tab_figures:
    st.markdown('<div class="section-title">Paper Figures</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">SVG figures generated from the frozen Groq benchmark summary.</div>',
        unsafe_allow_html=True,
    )
    if not figure_paths:
        st.warning("No figure files were found in the canonical summary directory.")
    for figure_path in figure_paths:
        _render_svg(figure_path, figure_path.name.replace("_", " ").replace(".svg", "").title())

with tab_notes:
    st.markdown('<div class="section-title">Method and Artifact Notes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Quick reminders for the paper, poster, and project presentation.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Figure notes**")
        st.markdown(notes.get("figure_notes", "_No figure notes found._"))
    with right:
        st.markdown("**Table notes**")
        st.markdown(notes.get("table_notes", "_No table notes found._"))
