from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents.langgraph_runtime import langgraph_available
from src.config.settings import Settings, load_settings
from src.eval.runner import run_experiment
from src.utils.io import read_jsonl


def canonical_summary_dir(settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    return settings.project_code_dir / "results" / "groq_full_summary"


def run_demo(
    task_split: str,
    system_variant: str,
    attack_mode: str | None = None,
    task_limit: int = 5,
    use_langgraph: bool = False,
    persist_traces: bool = True,
) -> dict:
    metrics, traces = run_experiment(
        task_split=task_split,
        system_variant=system_variant,
        attack_mode=attack_mode,
        task_limit=task_limit,
        use_langgraph=use_langgraph,
        persist_traces=persist_traces,
    )
    return {
        "metrics": metrics,
        "traces": [trace.to_dict() for trace in traces],
        "sample_trace": traces[0].to_dict() if traces else {},
    }


def load_summary_tables(summary_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    summary_dir = summary_dir or canonical_summary_dir()
    paths = {
        "summary_by_variant": summary_dir / "summary_by_variant.csv",
        "summary_by_attack_mode": summary_dir / "summary_by_attack_mode.csv",
        "main_table": summary_dir / "tables" / "table_1_main_results.csv",
        "attack_success_table": summary_dir / "tables" / "table_2_attack_mode_success.csv",
        "contamination_table": summary_dir / "tables" / "table_3_attack_mode_contamination.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        if path.exists():
            frames[key] = pd.read_csv(path)
    return frames


def load_figure_paths(summary_dir: Path | None = None) -> list[Path]:
    summary_dir = summary_dir or canonical_summary_dir()
    figures_dir = summary_dir / "figures"
    preferred = [
        "figure_1_architecture.svg",
        "figure_2_attacked_overview.svg",
        "figure_3_clean_vs_attacked_success.svg",
        "figure_4_attack_mode_success_heatmap.svg",
        "figure_5_attack_mode_contamination_heatmap.svg",
    ]
    return [figures_dir / name for name in preferred if (figures_dir / name).exists()]


def available_attack_modes(settings: Settings | None = None) -> list[str]:
    settings = settings or load_settings()
    records = read_jsonl(settings.attacks_dir / "attack_catalog.jsonl")
    return sorted({record["attack_mode"] for record in records})


def runtime_badges(settings: Settings | None = None) -> dict[str, str | bool]:
    settings = settings or load_settings()
    return {
        "provider": settings.provider,
        "model": settings.groq_model if settings.provider == "groq" else settings.ollama_model,
        "langgraph_available": langgraph_available(),
        "summary_dir": str(canonical_summary_dir(settings)),
    }


def extract_itinerary_frames(trace: dict) -> dict[str, pd.DataFrame]:
    sections: dict[str, list[dict]] = {"flight": [], "hotel": [], "attractions": []}
    for step in trace.get("final_itinerary", {}).get("itinerary_steps", []):
        step_type = step.get("type")
        selection = step.get("selection")
        if step_type == "attractions" and isinstance(selection, list):
            sections["attractions"] = selection
        elif step_type in {"flight", "hotel"} and isinstance(selection, dict):
            sections[step_type] = [selection]
    return {
        key: pd.DataFrame(rows)
        for key, rows in sections.items()
        if rows
    }


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""
