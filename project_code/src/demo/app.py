from __future__ import annotations

from src.eval.runner import run_experiment


def run_demo(task_split: str, system_variant: str, attack_mode: str | None = None) -> dict:
    metrics, traces = run_experiment(
        task_split=task_split,
        system_variant=system_variant,
        attack_mode=attack_mode,
        task_limit=5,
    )
    return {
        "metrics": metrics,
        "sample_trace": traces[0].to_dict() if traces else {},
    }
