from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANT_ORDER = [
    "single_agent_tool_use",
    "naive_multi_agent_shared_memory",
    "trust_aware_multi_agent",
    "ablation_no_provenance",
    "ablation_no_quarantine",
    "ablation_no_verifier",
]

VARIANT_LABELS = {
    "single_agent_tool_use": "Single Agent",
    "naive_multi_agent_shared_memory": "Naive Multi-Agent",
    "trust_aware_multi_agent": "Trust-Aware",
    "ablation_no_provenance": "No Provenance",
    "ablation_no_quarantine": "No Quarantine",
    "ablation_no_verifier": "No Verifier",
}

ATTACK_ORDER = [
    "conflicting_duplicate_record",
    "conflicting_schedule",
    "contaminated_tool_output",
    "dropped_field",
    "misleading_summary",
    "stale_availability",
    "stale_price",
]

ATTACK_LABELS = {
    "conflicting_duplicate_record": "Conflicting Duplicate Record",
    "conflicting_schedule": "Conflicting Schedule",
    "contaminated_tool_output": "Contaminated Tool Output",
    "dropped_field": "Dropped Field",
    "misleading_summary": "Misleading Summary",
    "stale_availability": "Stale Availability",
    "stale_price": "Stale Price",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value * 100:.1f}%"


def _format_latency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.1f}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _write_table_bundle(
    df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    caption: str,
    label: str,
) -> dict[str, str]:
    files = {
        f"{stem}.csv": df.to_csv(index=False),
        f"{stem}.md": _dataframe_to_markdown(df),
        f"{stem}.html": df.to_html(index=False, border=0),
        f"{stem}.tex": df.to_latex(index=False, escape=True, caption=caption, label=label),
    }
    written: dict[str, str] = {}
    for filename, content in files.items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        written[filename] = str(path)
    return written


def build_main_results_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    clean = (
        summary_df[summary_df["task_split"] == "clean_eval_tasks"][["system_variant", "task_success"]]
        .rename(columns={"task_success": "clean_success"})
        .set_index("system_variant")
    )
    attacked = (
        summary_df[summary_df["task_split"] == "attacked_eval_tasks"][
            [
                "system_variant",
                "task_success",
                "attack_success",
                "contamination_spread",
                "recovery_rate",
                "latency_ms",
            ]
        ]
        .rename(columns={"task_success": "attacked_success"})
        .set_index("system_variant")
    )

    rows: list[dict[str, str]] = []
    for variant in VARIANT_ORDER:
        clean_row = clean.loc[variant] if variant in clean.index else None
        attacked_row = attacked.loc[variant] if variant in attacked.index else None
        rows.append(
            {
                "System": VARIANT_LABELS[variant],
                "Clean Success": _format_percent(clean_row["clean_success"] if clean_row is not None else None),
                "Attacked Success": _format_percent(attacked_row["attacked_success"] if attacked_row is not None else None),
                "Attack Success": _format_percent(attacked_row["attack_success"] if attacked_row is not None else None),
                "Contamination": _format_percent(attacked_row["contamination_spread"] if attacked_row is not None else None),
                "Recovery": _format_percent(attacked_row["recovery_rate"] if attacked_row is not None else None),
                "Latency (ms)": _format_latency(attacked_row["latency_ms"] if attacked_row is not None else None),
            }
        )
    return pd.DataFrame(rows)


def build_attack_metric_df(attack_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = (
        attack_df.pivot(index="attack_mode", columns="system_variant", values=metric)
        .reindex(index=ATTACK_ORDER, columns=VARIANT_ORDER)
    )
    output = pd.DataFrame({"Attack Mode": [_attack_label(mode) for mode in ATTACK_ORDER]})
    for variant in VARIANT_ORDER:
        output[VARIANT_LABELS[variant]] = pivot[variant].map(_format_percent)
    return output


def _attack_label(mode: str) -> str:
    return ATTACK_LABELS.get(mode, mode.replace("_", " ").title())


def build_table_notes() -> str:
    return "\n".join(
        [
            "# Table Notes",
            "",
            "These tables are generated from the frozen Groq benchmark summary using pandas DataFrames.",
            "",
            "## Included tables",
            "",
            "- Table 1: main clean-vs-attacked benchmark comparison.",
            "- Table 2: attack-mode success breakdown.",
            "- Table 3: attack-mode contamination breakdown.",
            "",
            "## Export formats",
            "",
            "- CSV for downstream analysis",
            "- Markdown for quick insertion into notes",
            "- HTML for browser preview",
            "- LaTeX for the final paper",
            "",
        ]
    ) + "\n"


def export_paper_tables(summary_dir: Path) -> dict[str, str]:
    summary_df = pd.read_csv(summary_dir / "summary_by_variant.csv")
    attack_df = pd.read_csv(summary_dir / "summary_by_attack_mode.csv")
    output_dir = _ensure_dir(summary_dir / "tables")

    main_df = build_main_results_df(summary_df)
    success_df = build_attack_metric_df(attack_df, "task_success")
    contamination_df = build_attack_metric_df(attack_df, "contamination_spread")

    written: dict[str, str] = {}
    written.update(
        _write_table_bundle(
            main_df,
            output_dir,
            "table_1_main_results",
            caption="Main benchmark comparison on clean and attacked tasks.",
            label="tab:main_results",
        )
    )
    written.update(
        _write_table_bundle(
            success_df,
            output_dir,
            "table_2_attack_mode_success",
            caption="Attacked-task success by attack mode and system variant.",
            label="tab:attack_success_by_mode",
        )
    )
    written.update(
        _write_table_bundle(
            contamination_df,
            output_dir,
            "table_3_attack_mode_contamination",
            caption="Contamination spread by attack mode and system variant.",
            label="tab:contamination_by_mode",
        )
    )

    notes_path = output_dir / "table_notes.md"
    notes_path.write_text(build_table_notes(), encoding="utf-8")
    written["table_notes.md"] = str(notes_path)

    manifest_path = output_dir / "tables_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "summary_dir": str(summary_dir),
                "output_dir": str(output_dir),
                "backend": "pandas",
                "generated_files": written,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written["tables_manifest.json"] = str(manifest_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper tables from experiment summaries using pandas DataFrames.")
    parser.add_argument(
        "--summary-dir",
        default="project_code/results/groq_full_summary",
        help="Directory containing summary_by_variant.csv and summary_by_attack_mode.csv",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    written = export_paper_tables(summary_dir)
    print(f"Generated {len(written)} table artifacts in {summary_dir / 'tables'}")


if __name__ == "__main__":
    main()
