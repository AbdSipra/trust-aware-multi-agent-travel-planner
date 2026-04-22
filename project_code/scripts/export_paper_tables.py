from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

PALETTE = {
    "navy": "#16324f",
    "teal": "#1f7a8c",
    "gold": "#d9a441",
    "ink": "#1e2530",
    "muted": "#5f6c7b",
    "grid": "#d7dde5",
    "panel": "#f7f9fc",
    "white": "#ffffff",
    "trust": "#e8f5ef",
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

    written.update(_write_table_figure_bundle(df=df, output_dir=output_dir, stem=stem))
    return written


def _table_figure_size(df: pd.DataFrame) -> tuple[float, float]:
    width = max(9.5, len(df.columns) * 2.0)
    height = max(2.8, 1.5 + len(df) * 0.58)
    return width, height


def _write_table_figure_bundle(df: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, str]:
    fig = _build_table_figure(df=df, stem=stem)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        f"{stem}.svg": str(svg_path),
        f"{stem}.png": str(png_path),
    }


def _build_table_figure(df: pd.DataFrame, stem: str) -> plt.Figure:
    width, height = _table_figure_size(df)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    title_map = {
        "table_1_main_results": "Main Benchmark Comparison",
        "table_2_attack_mode_success": "Attack-Mode Success Breakdown",
        "table_3_attack_mode_contamination": "Attack-Mode Contamination Breakdown",
    }
    subtitle_map = {
        "table_1_main_results": "Poster-friendly summary of clean and attacked benchmark performance",
        "table_2_attack_mode_success": "Task success by attack mode and system variant",
        "table_3_attack_mode_contamination": "Contamination spread by attack mode and system variant",
    }

    fig.text(
        0.02,
        0.96,
        title_map.get(stem, stem.replace("_", " ").title()),
        fontsize=18,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.915,
        subtitle_map.get(stem, ""),
        fontsize=10.5,
        color=PALETTE["muted"],
        ha="left",
        va="top",
    )

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.02, 0.96, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(PALETTE["navy"])
            cell.set_text_props(color=PALETTE["white"], weight="bold")
            cell.set_height(cell.get_height() * 1.15)
            continue

        row_label = str(df.iloc[row - 1, 0]) if row - 1 < len(df.index) else ""
        if row_label == "Trust-Aware":
            cell.set_facecolor(PALETTE["trust"])
            cell.set_text_props(color=PALETTE["ink"], weight="bold")
        else:
            cell.set_facecolor(PALETTE["white"] if row % 2 else PALETTE["panel"])
            cell.set_text_props(color=PALETTE["ink"])

        if col == 0:
            cell.set_text_props(ha="left")

    return fig


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
            "- PNG for direct poster insertion",
            "- SVG for high-resolution vector placement",
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


def export_paper_table_images(
    summary_dir: Path,
    output_dir: Path | None = None,
    stem_suffix: str = "_poster",
) -> dict[str, str]:
    summary_df = pd.read_csv(summary_dir / "summary_by_variant.csv")
    attack_df = pd.read_csv(summary_dir / "summary_by_attack_mode.csv")
    output_dir = _ensure_dir(output_dir or (summary_dir / "tables"))

    frames = {
        f"table_1_main_results{stem_suffix}": build_main_results_df(summary_df),
        f"table_2_attack_mode_success{stem_suffix}": build_attack_metric_df(attack_df, "task_success"),
        f"table_3_attack_mode_contamination{stem_suffix}": build_attack_metric_df(
            attack_df, "contamination_spread"
        ),
    }

    written: dict[str, str] = {}
    for stem, df in frames.items():
        written.update(_write_table_figure_bundle(df=df, output_dir=output_dir, stem=stem))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper tables from experiment summaries using pandas DataFrames.")
    parser.add_argument(
        "--summary-dir",
        default="project_code/results/groq_full_summary",
        help="Directory containing summary_by_variant.csv and summary_by_attack_mode.csv",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Write only poster-friendly table images instead of the full text bundle.",
    )
    parser.add_argument(
        "--image-suffix",
        default="",
        help="Suffix for image-only exports, for example _poster.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for image-only exports.",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    if args.images_only:
        suffix = args.image_suffix or "_poster"
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        written = export_paper_table_images(summary_dir, output_dir=output_dir, stem_suffix=suffix)
        destination = output_dir or (summary_dir / "tables")
        print(f"Generated {len(written)} table image artifacts in {destination}")
    else:
        written = export_paper_tables(summary_dir)
        print(f"Generated {len(written)} table artifacts in {summary_dir / 'tables'}")


if __name__ == "__main__":
    main()
