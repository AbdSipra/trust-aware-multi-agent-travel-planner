from __future__ import annotations

import csv
import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "project_code" / "scripts" / "export_paper_visualizations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_paper_visualizations", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PaperVisualizationsTests(unittest.TestCase):
    def test_export_paper_visualizations_writes_svg_outputs(self) -> None:
        module = _load_module()
        temp_dir = PROJECT_ROOT / f".tmp_paper_visualizations_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (temp_dir / "summary_by_variant.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
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
                        "verifier_intervention_count",
                        "replan_count",
                        "tool_call_count",
                        "latency_ms",
                        "defensive_intervention",
                        "corrected_attack_target",
                    ]
                )
                rows = [
                    ["clean_eval_tasks", "single_agent_tool_use", "groq", "llama", "20", "1.0", "1.0", "0.0", "0.0", "0.0", "1.0", "0.0", "3.0", "500.0", "0.0", "0.0"],
                    ["clean_eval_tasks", "naive_multi_agent_shared_memory", "groq", "llama", "20", "1.0", "1.0", "0.0", "0.0", "0.0", "1.0", "0.0", "3.0", "510.0", "0.0", "0.0"],
                    ["clean_eval_tasks", "trust_aware_multi_agent", "groq", "llama", "20", "1.0", "1.0", "0.0", "0.0", "0.0", "1.0", "0.0", "3.0", "520.0", "0.0", "0.0"],
                    ["attacked_eval_tasks", "single_agent_tool_use", "groq", "llama", "20", "0.7", "0.7", "0.8", "0.5", "0.0", "1.0", "0.0", "3.0", "480.0", "0.0", "0.0"],
                    ["attacked_eval_tasks", "naive_multi_agent_shared_memory", "groq", "llama", "20", "0.7", "0.7", "1.0", "1.0", "0.0", "1.0", "0.0", "3.0", "530.0", "0.0", "0.0"],
                    ["attacked_eval_tasks", "trust_aware_multi_agent", "groq", "llama", "20", "1.0", "1.0", "0.0", "0.0", "1.0", "1.0", "0.0", "3.0", "550.0", "1.0", "0.5"],
                ]
                writer.writerows(rows)

            with (temp_dir / "summary_by_attack_mode.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "system_variant",
                        "attack_mode",
                        "model_provider",
                        "model_name",
                        "task_count",
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
                )
                writer.writerow(
                    ["trust_aware_multi_agent", "stale_price", "groq", "llama", "3", "1.0", "1.0", "0.0", "0.0", "1.0", "1.0", "0.0", "3.0", "500.0", "1.0", "1.0"]
                )
                writer.writerow(
                    ["single_agent_tool_use", "stale_price", "groq", "llama", "3", "1.0", "1.0", "1.0", "1.0", "0.0", "1.0", "0.0", "3.0", "450.0", "0.0", "0.0"]
                )

            written = module.export_paper_visualizations(temp_dir)
            self.assertIn("figure_2_attacked_overview.svg", written)
            self.assertTrue(Path(written["figure_2_attacked_overview.svg"]).exists())
            self.assertTrue(Path(written["figure_5_attack_mode_contamination_heatmap.svg"]).exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
