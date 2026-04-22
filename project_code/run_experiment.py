from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run travel-planner experiments.")
    parser.add_argument("--task-split", required=True, choices=["dev_tasks", "clean_eval_tasks", "attacked_eval_tasks"])
    parser.add_argument(
        "--system-variant",
        required=True,
        choices=[
            "single_agent_tool_use",
            "naive_multi_agent_shared_memory",
            "trust_aware_multi_agent",
            "ablation_no_quarantine",
            "ablation_no_verifier",
            "ablation_no_provenance",
        ],
    )
    parser.add_argument("--attack-mode", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--use-langgraph", action="store_true")
    parser.add_argument("--no-save-traces", action="store_true")
    args = parser.parse_args()

    metrics, traces = run_experiment(
        task_split=args.task_split,
        system_variant=args.system_variant,
        attack_mode=args.attack_mode,
        seed=args.seed,
        task_limit=args.task_limit,
        persist_traces=not args.no_save_traces,
        use_langgraph=args.use_langgraph,
    )
    print(json.dumps({"metrics": metrics, "run_count": len(traces)}, indent=2))


if __name__ == "__main__":
    main()
