from __future__ import annotations

import csv
import importlib.util
import unittest
from pathlib import Path
import uuid
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "project_code" / "scripts" / "export_paper_tables.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_paper_tables", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PaperTablesTests(unittest.TestCase):
    def test_export_paper_tables_writes_markdown_and_latex_outputs(self) -> None:
        module = _load_module()
        temp_dir = PROJECT_ROOT / f".tmp_paper_tables_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary_dir = temp_dir
            with (summary_dir / "summary_by_variant.csv").open("w", encoding="utf-8", newline="") as handle:
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
                writer.writerow(
                    [
                        "clean_eval_tasks",
                        "single_agent_tool_use",
                        "groq",
                        "llama",
                        "20",
                        "1.0",
                        "1.0",
                        "0.0",
                        "0.0",
                        "0.0",
                        "1.0",
                        "0.0",
                        "3.0",
                        "500.0",
                        "0.0",
                        "0.0",
                    ]
                )
                writer.writerow(
                    [
                        "attacked_eval_tasks",
                        "trust_aware_multi_agent",
                        "groq",
                        "llama",
                        "20",
                        "1.0",
                        "1.0",
                        "0.0",
                        "0.0",
                        "1.0",
                        "1.0",
                        "0.0",
                        "3.0",
                        "550.0",
                        "1.0",
                        "0.5",
                    ]
                )

            with (summary_dir / "summary_by_attack_mode.csv").open("w", encoding="utf-8", newline="") as handle:
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
                    [
                        "trust_aware_multi_agent",
                        "stale_price",
                        "groq",
                        "llama",
                        "3",
                        "1.0",
                        "1.0",
                        "0.0",
                        "0.0",
                        "1.0",
                        "1.0",
                        "0.0",
                        "3.0",
                        "500.0",
                        "1.0",
                        "1.0",
                    ]
                )

            written = module.export_paper_tables(summary_dir)

            main_md = Path(written["table_1_main_results.md"]).read_text(encoding="utf-8")
            success_md = Path(written["table_2_attack_mode_success.md"]).read_text(encoding="utf-8")
            main_tex = Path(written["table_1_main_results.tex"]).read_text(encoding="utf-8")

            self.assertIn("Trust-Aware", main_md)
            self.assertIn("100.0%", main_md)
            self.assertIn("Stale Price", success_md)
            self.assertIn(r"\caption{Main benchmark comparison on clean and attacked tasks.}", main_tex)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
