from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_CODE_DIR))

from src.eval.runner import run_experiment


def ensure_seed_data() -> None:
    knowledge_path = PROJECT_CODE_DIR / "data" / "knowledge" / "cities.json"
    if knowledge_path.exists():
        return
    subprocess.run([sys.executable, str(PROJECT_CODE_DIR / "scripts" / "generate_seed_data.py")], check=True)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_seed_data()
        cls._original_provider = os.environ.get("AGENTIC_MODEL_PROVIDER")
        os.environ["AGENTIC_MODEL_PROVIDER"] = "none"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_provider is None:
            os.environ.pop("AGENTIC_MODEL_PROVIDER", None)
        else:
            os.environ["AGENTIC_MODEL_PROVIDER"] = cls._original_provider

    def test_single_agent_dev_split_runs(self) -> None:
        metrics, traces = run_experiment(
            task_split="dev_tasks",
            system_variant="single_agent_tool_use",
            task_limit=3,
        )
        self.assertEqual(len(traces), 3)
        self.assertIn("task_success", metrics)

    def test_trust_aware_attacked_split_runs(self) -> None:
        metrics, traces = run_experiment(
            task_split="attacked_eval_tasks",
            system_variant="trust_aware_multi_agent",
            task_limit=3,
        )
        self.assertEqual(len(traces), 3)
        self.assertIn("contamination_spread", metrics)


if __name__ == "__main__":
    unittest.main()
