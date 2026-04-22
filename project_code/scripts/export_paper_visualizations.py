from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "navy": "#16324f",
    "teal": "#1f7a8c",
    "gold": "#d9a441",
    "red": "#bb4d3e",
    "green": "#3d7a57",
    "ink": "#1e2530",
    "muted": "#6d7885",
    "grid": "#d7dde5",
    "panel": "#f5f7fa",
    "white": "#ffffff",
    "sand": "#f0e3c2",
}

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


def _variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " ").title())


def _attack_label(mode: str) -> str:
    return ATTACK_LABELS.get(mode, mode.replace("_", " ").title())


def _read_summary(summary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_csv(summary_dir / "summary_by_variant.csv")
    attack_df = pd.read_csv(summary_dir / "summary_by_attack_mode.csv")
    return summary_df, attack_df


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    fill: str,
    text_color: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=(
            fill
            if fill not in {PALETTE["panel"], "#eef6f8", "#fff2eb"}
            else PALETTE["grid"]
        ),
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=text_color,
    )


def _add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    style: str = "->",
) -> None:
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=16, linewidth=1.8, color=color
    )
    ax.add_patch(arrow)


def build_architecture_figure(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.05,
        0.94,
        "Trust-Aware Multi-Agent Travel Planner",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.05,
        0.905,
        "Execution flow used in the final Groq benchmark",
        fontsize=11,
        color=PALETTE["muted"],
    )

    _add_box(
        ax,
        (0.05, 0.68),
        0.16,
        0.10,
        "User Task /\nTaskSpec",
        PALETTE["navy"],
        PALETTE["white"],
    )
    _add_box(
        ax, (0.30, 0.72), 0.18, 0.09, "Planner Agent", PALETTE["teal"], PALETTE["white"]
    )
    _add_box(
        ax, (0.30, 0.57), 0.18, 0.09, "Tool Agent", PALETTE["teal"], PALETTE["white"]
    )
    _add_box(
        ax, (0.30, 0.42), 0.18, 0.09, "Trust Governor", PALETTE["gold"], PALETTE["ink"]
    )
    _add_box(
        ax, (0.30, 0.27), 0.18, 0.09, "Verifier", PALETTE["green"], PALETTE["white"]
    )
    _add_box(
        ax,
        (0.30, 0.10),
        0.18,
        0.09,
        "Final Itinerary",
        PALETTE["navy"],
        PALETTE["white"],
    )

    _add_box(
        ax,
        (0.60, 0.56),
        0.18,
        0.14,
        "Structured Tools\nFlight / Hotel /\nAttraction Search",
        PALETTE["panel"],
        PALETTE["ink"],
    )
    _add_box(
        ax, (0.60, 0.39), 0.18, 0.09, "Shared Memory", PALETTE["panel"], PALETTE["ink"]
    )
    _add_box(
        ax, (0.83, 0.39), 0.14, 0.09, "Quarantine\nQueue", "#fff2eb", PALETTE["ink"]
    )
    _add_box(
        ax,
        (0.83, 0.57),
        0.14,
        0.11,
        "Reverification /\nTrusted Restore",
        "#eef6f8",
        PALETTE["ink"],
    )

    _add_arrow(ax, (0.21, 0.73), (0.30, 0.76), PALETTE["ink"])
    _add_arrow(ax, (0.39, 0.72), (0.39, 0.66), PALETTE["ink"])
    _add_arrow(ax, (0.48, 0.615), (0.60, 0.63), PALETTE["ink"])
    _add_arrow(ax, (0.69, 0.56), (0.69, 0.48), PALETTE["ink"])
    _add_arrow(ax, (0.48, 0.465), (0.60, 0.435), PALETTE["ink"])
    _add_arrow(ax, (0.78, 0.435), (0.83, 0.435), PALETTE["red"])
    _add_arrow(ax, (0.90, 0.48), (0.90, 0.57), PALETTE["teal"])
    _add_arrow(ax, (0.83, 0.625), (0.78, 0.625), PALETTE["teal"])
    _add_arrow(ax, (0.60, 0.435), (0.48, 0.315), PALETTE["ink"])
    _add_arrow(ax, (0.39, 0.27), (0.39, 0.19), PALETTE["ink"])

    notes = [
        "1. Planner issues structured search requests.",
        "2. Tool Agent retrieves source-grounded travel records.",
        "3. Trust Governor checks provenance and attack context.",
        "4. Suspicious critical records can be restored through trusted reverification.",
        "5. Verifier checks budget and calendar constraints before finalization.",
    ]
    for idx, note in enumerate(notes):
        fig.text(0.58, 0.86 - idx * 0.035, note, fontsize=10, color=PALETTE["ink"])

    return _save_figure(fig, output_dir / "figure_1_architecture.svg")


def build_attacked_overview_figure(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    attacked = summary_df[summary_df["task_split"] == "attacked_eval_tasks"].copy()
    attacked["variant_label"] = pd.Categorical(
        attacked["system_variant"], VARIANT_ORDER, ordered=True
    )
    attacked = attacked.sort_values("variant_label")

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["panel"])

    metrics = [
        ("task_success", PALETTE["navy"], "Task success"),
        ("contamination_spread", PALETTE["red"], "Contamination spread"),
        ("recovery_rate", PALETTE["teal"], "Recovery rate"),
    ]
    x = np.arange(len(attacked))
    width = 0.22

    for idx, (metric, color, label) in enumerate(metrics):
        offsets = x + (idx - 1) * width
        bars = ax.bar(
            offsets,
            attacked[metric].astype(float),
            width=width,
            label=label,
            color=color,
        )
        for bar, value in zip(bars, attacked[metric].astype(float), strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(
        "Figure 2. Attacked-Task Overview",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.0,
        1.03,
        "Groq full benchmark: success, contamination, and recovery across attacked tasks",
        transform=ax.transAxes,
        fontsize=10,
        color=PALETTE["muted"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_variant_label(variant) for variant in attacked["system_variant"]],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    return _save_figure(fig, output_dir / "figure_2_attacked_overview.svg")


def build_clean_vs_attacked_figure(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    focus_variants = [
        "single_agent_tool_use",
        "naive_multi_agent_shared_memory",
        "trust_aware_multi_agent",
    ]
    filtered = summary_df[summary_df["system_variant"].isin(focus_variants)].copy()
    pivot = (
        filtered.pivot(
            index="system_variant", columns="task_split", values="task_success"
        )
        .reindex(focus_variants)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["panel"])

    x = np.arange(len(pivot))
    width = 0.28
    clean_bars = ax.bar(
        x - width / 2,
        pivot["clean_eval_tasks"],
        width=width,
        color=PALETTE["navy"],
        label="Clean tasks",
    )
    attacked_bars = ax.bar(
        x + width / 2,
        pivot["attacked_eval_tasks"],
        width=width,
        color=PALETTE["gold"],
        label="Attacked tasks",
    )

    for bars in (clean_bars, attacked_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(
        "Figure 3. Clean vs Attacked Success",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.0,
        1.03,
        "Clean performance remains intact while trust-aware screening closes the attacked-task gap",
        transform=ax.transAxes,
        fontsize=10,
        color=PALETTE["muted"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_variant_label(variant) for variant in pivot.index])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Task success")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    return _save_figure(fig, output_dir / "figure_3_clean_vs_attacked_success.svg")


def _heatmap_cmap(metric: str) -> LinearSegmentedColormap:
    if metric == "contamination_spread":
        colors = [PALETTE["green"], PALETTE["sand"], PALETTE["red"]]
    else:
        colors = [PALETTE["red"], PALETTE["sand"], PALETTE["green"]]
    return LinearSegmentedColormap.from_list(f"{metric}_cmap", colors)


def build_attack_heatmaps(
    attack_df: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    variants = VARIANT_ORDER
    attack_modes = ATTACK_ORDER

    def render(metric: str, title: str, subtitle: str, filename: str) -> Path:
        pivot = (
            attack_df.pivot(
                index="attack_mode", columns="system_variant", values=metric
            )
            .reindex(index=attack_modes, columns=variants)
            .astype(float)
            .fillna(0.0)
        )

        fig, ax = plt.subplots(figsize=(13.5, 7.2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor(PALETTE["panel"])
        im = ax.imshow(
            pivot.values, cmap=_heatmap_cmap(metric), vmin=0.0, vmax=1.0, aspect="auto"
        )

        ax.set_title(
            title, loc="left", fontsize=18, fontweight="bold", color=PALETTE["ink"]
        )
        ax.text(
            0.0,
            1.03,
            subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color=PALETTE["muted"],
        )
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(
            [_variant_label(variant) for variant in variants], rotation=20, ha="right"
        )
        ax.set_yticks(np.arange(len(attack_modes)))
        ax.set_yticklabels([_attack_label(mode) for mode in attack_modes])

        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iloc[row_idx, col_idx]
                text_color = (
                    PALETTE["white"]
                    if value <= 0.25 or value >= 0.75
                    else PALETTE["ink"]
                )
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold",
                )

        colorbar = fig.colorbar(im, ax=ax, shrink=0.88)
        colorbar.set_label("Metric value")

        return _save_figure(fig, output_dir / filename)

    return (
        render(
            metric="task_success",
            title="Figure 4. Attack-Mode Success Heatmap",
            subtitle="Task success by attack mode and system variant on the full Groq attacked benchmark",
            filename="figure_4_attack_mode_success_heatmap.svg",
        ),
        render(
            metric="contamination_spread",
            title="Figure 5. Attack-Mode Contamination Heatmap",
            subtitle="Contamination spread by attack mode and system variant on the full Groq attacked benchmark",
            filename="figure_5_attack_mode_contamination_heatmap.svg",
        ),
    )


def build_figure_notes(output_dir: Path, generated_files: list[Path]) -> Path:
    notes = [
        "# Figure Notes",
        "",
        "These figures were generated directly from `summary_by_variant.csv` and `summary_by_attack_mode.csv` using matplotlib.",
        "",
        "## Recommended use",
        "",
        "- Figure 1: system architecture diagram for the methodology section.",
        "- Figure 2: main attacked-benchmark comparison figure for the results section.",
        "- Figure 3: clean-vs-attacked comparison for the core robustness claim.",
        "- Figure 4: attack-mode success heatmap for fine-grained analysis.",
        "- Figure 5: attack-mode contamination heatmap for the safety argument.",
        "",
        "## Generated files",
        "",
    ]
    notes.extend(f"- `{path.name}`" for path in generated_files)
    path = output_dir / "figure_notes.md"
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    return path


def export_paper_visualizations(summary_dir: Path) -> dict[str, str]:
    output_dir = _ensure_dir(summary_dir / "figures")
    summary_df, attack_df = _read_summary(summary_dir)

    generated = [
        build_architecture_figure(output_dir),
        build_attacked_overview_figure(summary_df, output_dir),
        build_clean_vs_attacked_figure(summary_df, output_dir),
    ]
    generated.extend(build_attack_heatmaps(attack_df, output_dir))
    notes_path = build_figure_notes(output_dir, generated)

    manifest = {
        "summary_dir": str(summary_dir),
        "generated_files": [str(path) for path in generated],
        "notes_file": str(notes_path),
        "backend": "matplotlib",
    }
    manifest_path = output_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    written = {path.name: str(path) for path in generated}
    written["figure_notes.md"] = str(notes_path)
    written["figures_manifest.json"] = str(manifest_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create publication-ready matplotlib visualizations from experiment summary CSVs."
    )
    parser.add_argument(
        "--summary-dir",
        default="project_code/results/groq_full_summary",
        help="Directory containing summary_by_variant.csv and summary_by_attack_mode.csv",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    written = export_paper_visualizations(summary_dir)
    print(f"Generated {len(written) - 2} figures in {summary_dir / 'figures'}")


if __name__ == "__main__":
    main()
