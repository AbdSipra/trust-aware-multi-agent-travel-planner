from __future__ import annotations

import html
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

.app-body, .app-body p, .app-body li, .app-body label {
    color: #1e2530;
}

.guide-panel {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d7dde5;
    border-left: 6px solid #1f7a8c;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 18px rgba(30, 37, 48, 0.04);
}

.guide-panel strong {
    color: #16324f;
}

.metric-card {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d7dde5;
    border-radius: 18px;
    padding: 0.9rem 1rem;
    min-height: 110px;
    box-shadow: 0 8px 18px rgba(30, 37, 48, 0.05);
}

.metric-card.compact {
    min-height: 96px;
}

.metric-label {
    color: #5f6c7b;
    font-size: 0.86rem;
    font-weight: 600;
    margin-bottom: 0.45rem;
    line-height: 1.2;
}

.metric-value {
    color: #1e2530;
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.05;
}

.metric-value.compact {
    font-size: 1.7rem;
}

.subtle-panel {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #d7dde5;
    border-radius: 18px;
    padding: 0.9rem 1rem;
    box-shadow: 0 8px 18px rgba(30, 37, 48, 0.04);
}

.notes-card {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d7dde5;
    border-radius: 20px;
    padding: 1.1rem 1.2rem;
    min-height: 420px;
    box-shadow: 0 10px 24px rgba(30, 37, 48, 0.05);
}

.notes-card h2 {
    color: #16324f;
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 2rem;
    margin: 0 0 1rem 0;
}

.notes-card h3 {
    color: #1f7a8c;
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.2rem;
    margin: 1.15rem 0 0.45rem 0;
}

.notes-card p,
.notes-card li {
    color: #223043;
    font-size: 1rem;
    line-height: 1.65;
}

.notes-card ul {
    margin: 0.2rem 0 0.9rem 1.25rem;
}

.notes-card code {
    color: #0b6b45;
    background: #e7f7ef;
    border: 1px solid #b5e7cc;
    border-radius: 8px;
    padding: 0.1rem 0.35rem;
    font-size: 0.92rem;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #f5f7fb !important;
}

[data-testid="stSidebar"] .stCaption {
    color: #cfd7e3 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="base-input"] > div {
    background: #11151f !important;
    color: #ffffff !important;
    border: 1px solid #3c4657 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #ffffff !important;
}

[data-testid="stTabs"] button {
    color: #425166 !important;
    font-weight: 700 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #16324f !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] label {
    color: #1e2530 !important;
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


def _format_inline_markdown(text: str) -> str:
    parts = text.split("`")
    formatted: list[str] = []
    for index, part in enumerate(parts):
        escaped = html.escape(part)
        if index % 2 == 1:
            formatted.append(f"<code>{escaped}</code>")
        else:
            formatted.append(escaped)
    return "".join(formatted)


def _render_notes_card(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    chunks: list[str] = ['<div class="notes-card">']
    in_list = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_list:
                chunks.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            if in_list:
                chunks.append("</ul>")
                in_list = False
            chunks.append(f"<h2>{_format_inline_markdown(line[2:])}</h2>")
            continue
        if line.startswith("## "):
            if in_list:
                chunks.append("</ul>")
                in_list = False
            chunks.append(f"<h3>{_format_inline_markdown(line[3:])}</h3>")
            continue
        if line.startswith("- "):
            if not in_list:
                chunks.append("<ul>")
                in_list = True
            chunks.append(f"<li>{_format_inline_markdown(line[2:])}</li>")
            continue
        if in_list:
            chunks.append("</ul>")
            in_list = False
        chunks.append(f"<p>{_format_inline_markdown(line)}</p>")

    if in_list:
        chunks.append("</ul>")
    chunks.append("</div>")
    return "".join(chunks)


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
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _meta_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card compact">
        <div class="metric-label">{label}</div>
        <div class="metric-value compact">{value}</div>
    </div>
    """


def _trace_option_label(trace: dict) -> str:
    task_id = trace.get("task_id", "unknown-task")
    attack = trace.get("attack_profile") or "no_attack"
    failure = trace.get("failure_reason") or "success"
    quarantine_count = len(trace.get("quarantine_events", []))
    return (
        f"{task_id} | attack={attack} | outcome={failure}"
        f" | quarantines={quarantine_count}"
    )


def _default_trace_index(traces: list[dict]) -> int:
    for index, trace in enumerate(traces):
        if trace.get("attack_profile"):
            return index
    return 0


def _trace_summary_frame(traces: list[dict]):
    import pandas as pd

    rows = []
    for trace in traces:
        metrics = trace.get("final_metrics", {})
        rows.append(
            {
                "task_id": trace.get("task_id"),
                "attack_profile": trace.get("attack_profile") or "none",
                "failure_reason": trace.get("failure_reason") or "",
                "quarantine_events": len(trace.get("quarantine_events", [])),
                "task_success": metrics.get("task_success"),
                "attack_success": metrics.get("attack_success"),
                "recovery_rate": metrics.get("recovery_rate"),
            }
        )
    return pd.DataFrame(rows)


def _run_guide(task_split: str) -> str:
    if task_split == "dev_tasks":
        return (
            "<strong>How this split works:</strong> `dev_tasks` is a clean tuning split. "
            "Every trace should show `attack_profile = none`. "
            "Use this split only to sanity-check planning behavior, not attack robustness."
        )
    if task_split == "clean_eval_tasks":
        return (
            "<strong>How this split works:</strong> `clean_eval_tasks` starts clean. "
            "If you choose an attack override, the system injects that attack onto matching clean base tasks. "
            "This is the best split for controlled single-attack demos."
        )
    return (
        "<strong>How this split works:</strong> `attacked_eval_tasks` already contains canonical attacked tasks. "
        "If you choose an attack override, it acts like a filter. "
        "Only matching tasks receive that attack, so some rows can still show `attack_profile = none`."
    )


def _show_trace(trace: dict) -> None:
    itinerary_frames = extract_itinerary_frames(trace)
    final_metrics = trace.get("final_metrics", {})

    st.markdown("**Selected trace**")
    meta_cols = st.columns(5)
    meta_cols[0].markdown(
        _meta_card("Task ID", str(trace.get("task_id", "--"))),
        unsafe_allow_html=True,
    )
    meta_cols[1].markdown(
        _meta_card("Attack Profile", str(trace.get("attack_profile") or "none")),
        unsafe_allow_html=True,
    )
    meta_cols[2].markdown(
        _meta_card("Failure Reason", str(trace.get("failure_reason") or "success")),
        unsafe_allow_html=True,
    )
    meta_cols[3].markdown(
        _meta_card("Quarantine Events", str(len(trace.get("quarantine_events", [])))),
        unsafe_allow_html=True,
    )
    meta_cols[4].markdown(
        _meta_card(
            "Task Success",
            (
                f"{float(final_metrics.get('task_success', 0.0)) * 100:.1f}%"
                if isinstance(final_metrics.get("task_success"), (int, float))
                else "--"
            ),
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "This panel reflects the exact task trace selected below, including the applied attack profile."
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown('<div class="section-title">Itinerary Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Structured selections chosen by the system for the selected run.</div>', unsafe_allow_html=True)
        if not itinerary_frames:
            st.info("No finalized itinerary was produced for this trace.")
        else:
            for section_name, frame in itinerary_frames.items():
                st.markdown(f"**{section_name.title()}**")
                st.dataframe(frame, use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="section-title">Trust and Verification Trail</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Agent messages, quarantine events, and verifier outputs for the selected run.</div>', unsafe_allow_html=True)
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


def _attack_override_help(task_split: str) -> tuple[str, bool]:
    if task_split == "dev_tasks":
        return (
            "Dev tasks are clean tuning tasks and do not have linked attack profiles. "
            "Attack override is disabled for this split.",
            True,
        )
    if task_split == "clean_eval_tasks":
        return (
            "For clean evaluation tasks, the selected attack will be mapped onto matching base tasks.",
            False,
        )
    return (
        "For attacked evaluation tasks, the selected attack acts as a filter. "
        "Only tasks whose canonical attack matches it will receive an attack; others will show attack_profile=none.",
        False,
    )


def _render_attack_coverage(traces: list[dict], requested_attack: str, task_split: str) -> None:
    if not requested_attack:
        return

    attacked_count = sum(1 for trace in traces if trace.get("attack_profile"))
    total_count = len(traces)

    if task_split == "dev_tasks":
        st.info(
            "No attacks were applied because `dev_tasks` is a clean split without linked attack mappings."
        )
        return

    if attacked_count == 0:
        st.warning(
            f"No traces received `{requested_attack}` in this run. "
            "Choose `clean_eval_tasks` to map that attack onto clean base tasks, "
            "or use a matching task subset under `attacked_eval_tasks`."
        )
        return

    if attacked_count < total_count:
        st.info(
            f"Applied `{requested_attack}` to {attacked_count}/{total_count} traces in this run. "
            "This is expected when the selected split includes tasks whose canonical attack does not match the chosen filter."
        )
        return

    st.success(f"Applied `{requested_attack}` to all {total_count} traces in this run.")


def _filter_display_traces(traces: list[dict], show_only_attacked: bool) -> list[dict]:
    if not show_only_attacked:
        return traces
    attacked = [trace for trace in traces if trace.get("attack_profile")]
    return attacked or traces


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
    attack_help, attack_disabled = _attack_override_help(task_split)
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
    if attack_disabled and st.session_state.get("attack_mode_widget"):
        st.session_state["attack_mode_widget"] = ""
    attack_mode = st.selectbox(
        "Attack override",
        [""] + _cached_attack_modes(),
        disabled=attack_disabled,
        help=attack_help,
        key="attack_mode_widget",
    )
    st.caption(attack_help)
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
    st.markdown('<div class="app-body">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live Experiment Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Run a small slice of the benchmark and inspect the trust-aware behavior step by step.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="guide-panel">{_run_guide(demo_meta.get("task_split", task_split))}</div>', unsafe_allow_html=True)

    if demo_result:
        st.markdown(
            f'<div class="small-note">Latest run: split={demo_meta.get("task_split")} | variant={demo_meta.get("system_variant")} | attack={demo_meta.get("attack_mode")} | LangGraph={"on" if demo_meta.get("use_langgraph") else "off"}</div>',
            unsafe_allow_html=True,
        )
        _metric_row(demo_result["metrics"])
        st.markdown("---")
        traces = demo_result.get("traces", [])
        if traces:
            _render_attack_coverage(
                traces=traces,
                requested_attack=str(demo_meta.get("attack_mode", "")),
                task_split=str(demo_meta.get("task_split", "")),
            )
            st.markdown('<div class="subtle-panel">Use the trace selector below to inspect a specific task, including its actual applied attack profile.</div>', unsafe_allow_html=True)
            attacked_count = sum(1 for trace in traces if trace.get("attack_profile"))
            show_only_attacked_default = bool(attacked_count and demo_meta.get("task_split") != "dev_tasks")
            show_only_attacked = st.checkbox(
                "Show only traces with applied attack",
                value=show_only_attacked_default,
                disabled=not bool(attacked_count),
                help="When enabled, the selector and table focus only on traces where an attack was actually applied.",
            )
            display_traces = _filter_display_traces(traces, show_only_attacked)
            trace_options = [_trace_option_label(trace) for trace in display_traces]
            selected_label = st.selectbox(
                "Inspect trace",
                trace_options,
                index=_default_trace_index(display_traces),
                help="Choose the exact task trace you want to inspect. Attack override only applies when a matching attack exists for that task.",
            )
            selected_trace = display_traces[trace_options.index(selected_label)]
            st.dataframe(
                _trace_summary_frame(display_traces),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("---")
            _show_trace(selected_trace)
        else:
            _show_trace(demo_result["sample_trace"])
    else:
        st.info("Run an experiment from the sidebar to populate this lab view.")
    st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="app-body">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Method and Artifact Notes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Quick reminders for the paper, poster, and project presentation.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown(_render_notes_card(notes.get("figure_notes", "# Figure Notes\n\nNo figure notes found.")), unsafe_allow_html=True)
    with right:
        st.markdown(_render_notes_card(notes.get("table_notes", "# Table Notes\n\nNo table notes found.")), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
