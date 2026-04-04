from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_CODE_DIR))

from src.config.settings import load_settings
from src.eval.metrics import compute_run_metrics
from src.state.schemas import RunTrace, TaskSpec
from src.utils.io import read_json, read_jsonl, write_csv, write_json

TASK_SPLITS = ("dev_tasks", "clean_eval_tasks", "attacked_eval_tasks")
METRIC_FIELDS = [
    "task_success",
    "hard_constraint_satisfaction",
    "attack_success",
    "contamination_spread",
    "recovery_rate",
    "verifier_intervention_count",
    "replan_count",
    "tool_call_count",
    "latency_ms",
    "defensive_intervention",
    "corrected_attack_target",
]


def _load_tasks() -> tuple[dict[str, TaskSpec], dict[str, str]]:
    settings = load_settings()
    tasks: dict[str, TaskSpec] = {}
    task_splits: dict[str, str] = {}
    for split in TASK_SPLITS:
        for record in read_jsonl(settings.tasks_dir / f"{split}.jsonl"):
            task = TaskSpec(**record)
            tasks[task.task_id] = task
            task_splits[task.task_id] = split
    return tasks, task_splits


def _load_attack_lookup() -> dict[str, dict]:
    settings = load_settings()
    return {
        record["attack_id"]: record
        for record in read_jsonl(settings.attacks_dir / "attack_catalog.jsonl")
    }


def _latest_trace_paths(runs_dir: Path, model_provider: str | None, model_name: str | None) -> list[Path]:
    latest_by_key: dict[tuple[str, str, str, str], Path] = {}
    for path in runs_dir.glob("*.json"):
        payload = read_json(path)
        provider = payload.get("model_provider", "")
        name = payload.get("model_name", "")
        if model_provider and provider != model_provider:
            continue
        if model_name and name != model_name:
            continue
        key = (
            payload.get("system_variant", ""),
            payload.get("task_id", ""),
            provider,
            name,
        )
        current = latest_by_key.get(key)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest_by_key[key] = path
    return sorted(latest_by_key.values(), key=lambda item: item.name)


def _aggregate(rows: list[dict], group_fields: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    summary_rows: list[dict] = []
    for key, members in sorted(grouped.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["task_count"] = len(members)
        for metric in METRIC_FIELDS:
            summary[metric] = round(sum(float(member[metric]) for member in members) / len(members), 4)
        summary_rows.append(summary)
    return summary_rows


def _markdown_table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reproducible experiment summaries from saved run traces.")
    parser.add_argument("--model-provider", default=None, help="Optional provider filter, e.g. groq or ollama.")
    parser.add_argument("--model-name", default=None, help="Optional exact model-name filter.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to project_code/results/latest_summary.",
    )
    parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Include dev-task runs in the exported summaries.",
    )
    args = parser.parse_args()

    settings = load_settings()
    output_dir = Path(args.output_dir) if args.output_dir else settings.project_code_dir / "results" / "latest_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks, task_splits = _load_tasks()
    attack_lookup = _load_attack_lookup()
    trace_paths = _latest_trace_paths(settings.runs_dir, args.model_provider, args.model_name)

    per_trace_rows: list[dict] = []
    for path in trace_paths:
        payload = read_json(path)
        trace = RunTrace(**payload)
        task = tasks.get(trace.task_id)
        if task is None:
            continue
        task_split = task_splits[task.task_id]
        if task_split == "dev_tasks" and not args.include_dev:
            continue
        attack_profile = attack_lookup.get(task.attack_id) if task.attack_id else None
        metrics = compute_run_metrics(trace, task=task, attack_profile=attack_profile)
        row = {
            "run_id": trace.run_id,
            "task_id": trace.task_id,
            "task_split": task_split,
            "model_provider": trace.model_provider,
            "model_name": trace.model_name,
            "system_variant": trace.system_variant,
            "attack_mode": attack_profile["attack_mode"] if attack_profile else "",
            "failure_reason": trace.failure_reason or "",
            "run_file": path.name,
        }
        row.update(metrics)
        per_trace_rows.append(row)

    summary_by_variant = _aggregate(
        per_trace_rows,
        ["task_split", "system_variant", "model_provider", "model_name"],
    )
    attacked_rows = [row for row in per_trace_rows if row["task_split"] == "attacked_eval_tasks"]
    summary_by_attack_mode = _aggregate(
        attacked_rows,
        ["system_variant", "attack_mode", "model_provider", "model_name"],
    )
    failure_cases = [
        row
        for row in per_trace_rows
        if row["task_success"] == 0 or row["contamination_spread"] == 1
    ]

    write_csv(
        output_dir / "per_trace_metrics.csv",
        per_trace_rows,
        [
            "run_id",
            "task_id",
            "task_split",
            "model_provider",
            "model_name",
            "system_variant",
            "attack_mode",
            "failure_reason",
            "run_file",
            *METRIC_FIELDS,
        ],
    )
    write_csv(
        output_dir / "summary_by_variant.csv",
        summary_by_variant,
        ["task_split", "system_variant", "model_provider", "model_name", "task_count", *METRIC_FIELDS],
    )
    write_csv(
        output_dir / "summary_by_attack_mode.csv",
        summary_by_attack_mode,
        ["system_variant", "attack_mode", "model_provider", "model_name", "task_count", *METRIC_FIELDS],
    )
    write_json(output_dir / "failure_cases.json", failure_cases)
    write_json(
        output_dir / "manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "model_provider": args.model_provider,
                "model_name": args.model_name,
                "include_dev": args.include_dev,
            },
            "trace_count": len(per_trace_rows),
            "source_runs_dir": str(settings.runs_dir),
            "source_run_files": [row["run_file"] for row in per_trace_rows],
        },
    )

    markdown = [
        "# Experiment Summary",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Trace count: {len(per_trace_rows)}",
        f"- Provider filter: {args.model_provider or 'all'}",
        f"- Model filter: {args.model_name or 'all'}",
        "",
        "## Summary by Variant",
        "",
        _markdown_table(
            summary_by_variant,
            [
                "task_split",
                "system_variant",
                "model_provider",
                "model_name",
                "task_count",
                "task_success",
                "hard_constraint_satisfaction",
                "attack_success",
                "contamination_spread",
                "recovery_rate",
            ],
        ),
        "",
        "## Attacked Breakdown by Attack Mode",
        "",
        _markdown_table(
            summary_by_attack_mode,
            [
                "system_variant",
                "attack_mode",
                "model_provider",
                "model_name",
                "task_count",
                "task_success",
                "contamination_spread",
                "recovery_rate",
            ],
        ),
        "",
        "## Failure Cases",
        "",
        _markdown_table(
            failure_cases,
            [
                "task_id",
                "task_split",
                "system_variant",
                "model_provider",
                "model_name",
                "attack_mode",
                "task_success",
                "contamination_spread",
                "failure_reason",
                "run_file",
            ],
        ),
    ]
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(f"Exported experiment summary to {output_dir}")


if __name__ == "__main__":
    main()
