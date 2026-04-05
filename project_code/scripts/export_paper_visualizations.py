from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_CODE_DIR))

from src.utils.io import write_json


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
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _svg_text(x: float, y: float, text: str, size: int = 16, weight: str = "400", fill: str | None = None, anchor: str = "start", rotate: int | None = None) -> str:
    fill_attr = fill or PALETTE["ink"]
    safe = html.escape(text)
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill_attr}" '
        f'text-anchor="{anchor}"{transform}>{safe}</text>'
    )


def _svg_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none", stroke_width: int = 0, rx: int = 0, opacity: float | None = None) -> str:
    opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}"{opacity_attr}/>'
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: int = 2, dash: str | None = None, marker_end: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker_end})"' if marker_end else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr}{marker_attr}/>'
    )


def _metric_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    if value >= 0.95:
        return "#0f6d3c"
    if value >= 0.75:
        return "#4f8f3d"
    if value >= 0.5:
        return "#b58b2a"
    if value >= 0.25:
        return "#c46737"
    return "#b33d3d"


def _contamination_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    if value <= 0.05:
        return "#0f6d3c"
    if value <= 0.25:
        return "#8a9836"
    if value <= 0.5:
        return "#ca7f2e"
    return "#b33d3d"


def _bar_height(value: float, max_height: float) -> float:
    return max(0.0, min(max_height, value * max_height))


def _write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M0,0 L0,6 L9,3 z" fill="{PALETTE["ink"]}"/>',
        "</marker>",
        "</defs>",
        _svg_rect(0, 0, width, height, PALETTE["white"]),
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _variant_label(variant: str) -> str:
    mapping = {
        "single_agent_tool_use": "Single Agent",
        "naive_multi_agent_shared_memory": "Naive Multi-Agent",
        "trust_aware_multi_agent": "Trust-Aware",
        "ablation_no_provenance": "No Provenance",
        "ablation_no_quarantine": "No Quarantine",
        "ablation_no_verifier": "No Verifier",
    }
    return mapping.get(variant, variant.replace("_", " ").title())


def _attack_label(mode: str) -> str:
    return mode.replace("_", " ").title()


def build_architecture_svg(output_dir: Path) -> Path:
    width, height = 1400, 840
    body: list[str] = [
        _svg_text(70, 70, "Trust-Aware Multi-Agent Travel Planner", size=28, weight="700"),
        _svg_text(70, 100, "Execution flow used in the final Groq benchmark", size=15, fill=PALETTE["muted"]),
    ]

    boxes = {
        "task": (80, 170, 240, 90, PALETTE["navy"], "User Task / TaskSpec"),
        "planner": (410, 150, 250, 90, PALETTE["teal"], "Planner Agent"),
        "tool_agent": (410, 290, 250, 90, PALETTE["teal"], "Tool Agent"),
        "trust": (410, 430, 250, 90, PALETTE["gold"], "Trust Governor"),
        "verifier": (410, 570, 250, 90, PALETTE["green"], "Verifier"),
        "final": (410, 710, 250, 90, PALETTE["navy"], "Final Itinerary"),
        "tools": (820, 270, 260, 150, PALETTE["panel"], "Structured Tools"),
        "memory": (820, 475, 260, 90, PALETTE["panel"], "Shared Memory"),
        "quarantine": (1120, 475, 220, 90, "#fff2eb", "Quarantine Queue"),
        "fallback": (1120, 270, 220, 110, "#eef6f8", "Reverification / Trusted Restore"),
    }

    for x, y, w, h, fill, label in boxes.values():
        stroke = PALETTE["grid"] if fill == PALETTE["panel"] else fill
        text_fill = PALETTE["ink"] if fill == PALETTE["panel"] or fill.startswith("#fff") or fill.startswith("#eef") else PALETTE["white"]
        body.append(_svg_rect(x, y, w, h, fill, stroke=stroke, stroke_width=2, rx=18))
        body.append(_svg_text(x + w / 2, y + h / 2 + 6, label, size=18, weight="600", fill=text_fill, anchor="middle"))

    body.extend(
        [
            _svg_line(320, 215, 410, 195, PALETTE["ink"], marker_end="arrow"),
            _svg_line(535, 240, 535, 290, PALETTE["ink"], marker_end="arrow"),
            _svg_line(660, 335, 820, 335, PALETTE["ink"], marker_end="arrow"),
            _svg_line(950, 420, 950, 475, PALETTE["ink"], marker_end="arrow"),
            _svg_line(660, 470, 820, 520, PALETTE["ink"], marker_end="arrow"),
            _svg_line(1080, 520, 1120, 520, PALETTE["red"], marker_end="arrow"),
            _svg_line(1230, 475, 1230, 380, PALETTE["teal"], marker_end="arrow"),
            _svg_line(1120, 335, 1080, 335, PALETTE["teal"], marker_end="arrow"),
            _svg_line(820, 520, 660, 615, PALETTE["ink"], marker_end="arrow"),
            _svg_line(535, 660, 535, 710, PALETTE["ink"], marker_end="arrow"),
        ]
    )

    body.extend(
        [
            _svg_text(1115, 316, "FlightSearchTool / HotelSearchTool / AttractionSearchTool", size=14, anchor="end", fill=PALETTE["muted"]),
            _svg_text(950, 505, "accepted observations", size=13, anchor="middle", fill=PALETTE["muted"]),
            _svg_text(1230, 505, "high-risk items", size=13, anchor="middle", fill=PALETTE["red"]),
            _svg_text(1230, 365, "recover critical records", size=13, anchor="middle", fill=PALETTE["teal"]),
            _svg_text(740, 605, "verified candidate", size=13, anchor="middle", fill=PALETTE["muted"]),
        ]
    )

    notes_y = 170
    for idx, note in enumerate(
        [
            "1. Planner issues structured search queries.",
            "2. Tool Agent retrieves local-source-grounded travel data.",
            "3. Trust Governor screens each observation with provenance and attack context.",
            "4. Accepted results enter shared memory; suspicious critical records can be restored via trusted reverification.",
            "5. Verifier checks calendar and budget constraints before finalizing the itinerary.",
        ]
    ):
        body.append(_svg_text(850, notes_y + idx * 24, note, size=14, fill=PALETTE["ink"]))

    path = output_dir / "figure_1_architecture.svg"
    _write_svg(path, width, height, body)
    return path


def build_attacked_overview_svg(summary_rows: list[dict[str, str]], output_dir: Path) -> Path:
    attacked = [row for row in summary_rows if row["task_split"] == "attacked_eval_tasks"]
    attacked.sort(key=lambda row: row["system_variant"])
    width, height = 1500, 860
    chart_x, chart_y = 110, 170
    chart_w, chart_h = 1260, 520
    body: list[str] = [
        _svg_text(70, 70, "Figure 2. Attacked-Task Overview", size=28, weight="700"),
        _svg_text(70, 100, "Groq full benchmark: success, contamination, and recovery across all attacked tasks", size=15, fill=PALETTE["muted"]),
        _svg_rect(60, 130, 1320, 610, PALETTE["panel"], stroke=PALETTE["grid"], stroke_width=1, rx=20),
    ]

    for step in range(6):
        y = chart_y + chart_h - step * (chart_h / 5)
        body.append(_svg_line(chart_x, y, chart_x + chart_w, y, PALETTE["grid"], stroke_width=1))
        value = step / 5
        body.append(_svg_text(chart_x - 18, y + 5, f"{value:.1f}", size=13, anchor="end", fill=PALETTE["muted"]))

    body.append(_svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, PALETTE["ink"], stroke_width=2))
    body.append(_svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, PALETTE["ink"], stroke_width=2))

    group_width = chart_w / max(1, len(attacked))
    metric_specs = [
        ("task_success", PALETTE["navy"], "Task success"),
        ("contamination_spread", PALETTE["red"], "Contamination spread"),
        ("recovery_rate", PALETTE["teal"], "Recovery rate"),
    ]

    for index, row in enumerate(attacked):
        x0 = chart_x + index * group_width
        bar_w = group_width * 0.18
        gap = group_width * 0.08
        label_x = x0 + group_width / 2
        for metric_idx, (metric, color, _) in enumerate(metric_specs):
            value = float(row[metric])
            bar_h = _bar_height(value, chart_h)
            bar_x = x0 + group_width * 0.18 + metric_idx * (bar_w + gap)
            bar_y = chart_y + chart_h - bar_h
            body.append(_svg_rect(bar_x, bar_y, bar_w, bar_h, color, rx=8))
            body.append(_svg_text(bar_x + bar_w / 2, bar_y - 10, f"{value:.2f}", size=12, anchor="middle", fill=PALETTE["ink"]))
        body.append(_svg_text(label_x, chart_y + chart_h + 32, _variant_label(row["system_variant"]), size=14, anchor="middle", rotate=-15))

    legend_x = 960
    legend_y = 760
    for idx, (_, color, label) in enumerate(metric_specs):
        body.append(_svg_rect(legend_x + idx * 170, legend_y, 20, 20, color, rx=4))
        body.append(_svg_text(legend_x + 28 + idx * 170, legend_y + 16, label, size=14))

    notes = [
        "Trust-Aware reaches perfect attacked-task success and zero contamination.",
        "No Provenance falls back to the naive attacked profile.",
        "No Quarantine preserves success but allows contamination to spread.",
    ]
    for idx, note in enumerate(notes):
        body.append(_svg_text(90, 780 + idx * 22, note, size=14, fill=PALETTE["ink"]))

    path = output_dir / "figure_2_attacked_overview.svg"
    _write_svg(path, width, height, body)
    return path


def build_clean_vs_attacked_svg(summary_rows: list[dict[str, str]], output_dir: Path) -> Path:
    focus_variants = [
        "single_agent_tool_use",
        "naive_multi_agent_shared_memory",
        "trust_aware_multi_agent",
    ]
    rows = [row for row in summary_rows if row["system_variant"] in focus_variants]
    rows.sort(key=lambda row: (focus_variants.index(row["system_variant"]), row["task_split"]))

    width, height = 1300, 760
    chart_x, chart_y = 120, 170
    chart_w, chart_h = 1060, 430
    body: list[str] = [
        _svg_text(70, 70, "Figure 3. Clean vs Attacked Success", size=28, weight="700"),
        _svg_text(70, 100, "Clean performance stays intact while the trust-aware method closes the attacked-task gap", size=15, fill=PALETTE["muted"]),
        _svg_rect(60, 130, 1180, 540, PALETTE["panel"], stroke=PALETTE["grid"], stroke_width=1, rx=20),
    ]

    for step in range(6):
        y = chart_y + chart_h - step * (chart_h / 5)
        body.append(_svg_line(chart_x, y, chart_x + chart_w, y, PALETTE["grid"], stroke_width=1))
        body.append(_svg_text(chart_x - 15, y + 5, f"{step / 5:.1f}", size=13, anchor="end", fill=PALETTE["muted"]))

    body.append(_svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, PALETTE["ink"], stroke_width=2))
    body.append(_svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, PALETTE["ink"], stroke_width=2))

    groups = {variant: {} for variant in focus_variants}
    for row in rows:
        groups[row["system_variant"]][row["task_split"]] = float(row["task_success"])

    group_width = chart_w / len(focus_variants)
    for index, variant in enumerate(focus_variants):
        x0 = chart_x + index * group_width
        clean = groups[variant].get("clean_eval_tasks", 0.0)
        attacked = groups[variant].get("attacked_eval_tasks", 0.0)
        clean_bar_h = _bar_height(clean, chart_h)
        attacked_bar_h = _bar_height(attacked, chart_h)
        body.append(_svg_rect(x0 + group_width * 0.2, chart_y + chart_h - clean_bar_h, group_width * 0.2, clean_bar_h, PALETTE["navy"], rx=10))
        body.append(_svg_rect(x0 + group_width * 0.48, chart_y + chart_h - attacked_bar_h, group_width * 0.2, attacked_bar_h, PALETTE["gold"], rx=10))
        body.append(_svg_text(x0 + group_width * 0.3, chart_y + chart_h - clean_bar_h - 10, f"{clean:.2f}", size=13, anchor="middle"))
        body.append(_svg_text(x0 + group_width * 0.58, chart_y + chart_h - attacked_bar_h - 10, f"{attacked:.2f}", size=13, anchor="middle"))
        body.append(_svg_text(x0 + group_width / 2, chart_y + chart_h + 35, _variant_label(variant), size=15, weight="600", anchor="middle"))

    legend_x = 770
    legend_y = 625
    body.append(_svg_rect(legend_x, legend_y, 20, 20, PALETTE["navy"], rx=4))
    body.append(_svg_text(legend_x + 30, legend_y + 16, "Clean tasks", size=14))
    body.append(_svg_rect(legend_x + 190, legend_y, 20, 20, PALETTE["gold"], rx=4))
    body.append(_svg_text(legend_x + 220, legend_y + 16, "Attacked tasks", size=14))

    path = output_dir / "figure_3_clean_vs_attacked_success.svg"
    _write_svg(path, width, height, body)
    return path


def build_attack_heatmaps(attack_rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    attack_modes = sorted({row["attack_mode"] for row in attack_rows})
    variants = [
        "single_agent_tool_use",
        "naive_multi_agent_shared_memory",
        "trust_aware_multi_agent",
        "ablation_no_provenance",
        "ablation_no_quarantine",
        "ablation_no_verifier",
    ]

    def render(metric: str, title: str, subtitle: str, color_fn, filename: str) -> Path:
        width, height = 1560, 880
        origin_x, origin_y = 250, 180
        cell_w, cell_h = 170, 70
        body: list[str] = [
            _svg_text(70, 70, title, size=28, weight="700"),
            _svg_text(70, 100, subtitle, size=15, fill=PALETTE["muted"]),
            _svg_rect(60, 130, 1440, 690, PALETTE["panel"], stroke=PALETTE["grid"], stroke_width=1, rx=20),
        ]

        for col_idx, variant in enumerate(variants):
            x = origin_x + col_idx * cell_w + cell_w / 2
            body.append(_svg_text(x, origin_y - 18, _variant_label(variant), size=14, weight="600", anchor="middle"))

        for row_idx, attack_mode in enumerate(attack_modes):
            y = origin_y + row_idx * cell_h + cell_h / 2 + 6
            body.append(_svg_text(origin_x - 20, y, _attack_label(attack_mode), size=14, weight="600", anchor="end"))
            for col_idx, variant in enumerate(variants):
                x = origin_x + col_idx * cell_w
                value = 0.0
                for row in attack_rows:
                    if row["attack_mode"] == attack_mode and row["system_variant"] == variant:
                        value = float(row[metric])
                        break
                body.append(_svg_rect(x, origin_y + row_idx * cell_h, cell_w - 8, cell_h - 8, color_fn(value), rx=10))
                body.append(_svg_text(x + (cell_w - 8) / 2, origin_y + row_idx * cell_h + 42, f"{value:.2f}", size=16, weight="700", fill=PALETTE["white"], anchor="middle"))

        legend_x = 250
        legend_y = 730
        for idx, value in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
            color = color_fn(value)
            body.append(_svg_rect(legend_x + idx * 120, legend_y, 42, 24, color, rx=5))
            body.append(_svg_text(legend_x + idx * 120 + 52, legend_y + 18, f"{value:.2f}", size=13))

        path = output_dir / filename
        _write_svg(path, width, height, body)
        return path

    return (
        render(
            metric="task_success",
            title="Figure 4. Attack-Mode Success Heatmap",
            subtitle="Task success by attack mode and system variant on the full Groq attacked benchmark",
            color_fn=_metric_color,
            filename="figure_4_attack_mode_success_heatmap.svg",
        ),
        render(
            metric="contamination_spread",
            title="Figure 5. Attack-Mode Contamination Heatmap",
            subtitle="Contamination spread by attack mode and system variant on the full Groq attacked benchmark",
            color_fn=_contamination_color,
            filename="figure_5_attack_mode_contamination_heatmap.svg",
        ),
    )


def build_figure_notes(output_dir: Path, generated_files: list[Path]) -> Path:
    notes = [
        "# Figure Notes",
        "",
        "These figures were generated directly from `summary_by_variant.csv` and `summary_by_attack_mode.csv` in the same summary directory.",
        "",
        "## Recommended use",
        "",
        "- Figure 1: system architecture diagram for the methodology section.",
        "- Figure 2: main attacked-benchmark comparison figure for the results section.",
        "- Figure 3: clean-vs-attacked comparison for the paper's main robustness claim.",
        "- Figure 4: attack-mode success heatmap for fine-grained analysis.",
        "- Figure 5: contamination heatmap for the safety argument.",
        "",
        "## Generated files",
        "",
    ]
    notes.extend(f"- `{path.name}`" for path in generated_files)
    path = output_dir / "figure_notes.md"
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready SVG visualizations from experiment summary CSVs.")
    parser.add_argument(
        "--summary-dir",
        default="project_code/results/groq_full_summary",
        help="Directory containing summary_by_variant.csv and summary_by_attack_mode.csv",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    output_dir = _ensure_dir(summary_dir / "figures")

    summary_rows = _read_csv(summary_dir / "summary_by_variant.csv")
    attack_rows = _read_csv(summary_dir / "summary_by_attack_mode.csv")

    generated = [
        build_architecture_svg(output_dir),
        build_attacked_overview_svg(summary_rows, output_dir),
        build_clean_vs_attacked_svg(summary_rows, output_dir),
    ]
    generated.extend(build_attack_heatmaps(attack_rows, output_dir))
    notes_path = build_figure_notes(output_dir, generated)

    manifest = {
        "summary_dir": str(summary_dir),
        "generated_files": [str(path) for path in generated],
        "notes_file": str(notes_path),
    }
    write_json(output_dir / "figures_manifest.json", manifest)
    print(f"Generated {len(generated)} SVG figures in {output_dir}")


if __name__ == "__main__":
    main()
