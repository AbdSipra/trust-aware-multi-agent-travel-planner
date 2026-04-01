from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_CODE_DIR))

from src.config.settings import load_settings
from src.tools.knowledge_store import KnowledgeStore
from src.tools.registry import ToolRegistry


def ensure_seed_data() -> None:
    knowledge_path = PROJECT_CODE_DIR / "data" / "knowledge" / "cities.json"
    if knowledge_path.exists():
        return
    subprocess.run([sys.executable, str(PROJECT_CODE_DIR / "scripts" / "generate_seed_data.py")], check=True)


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_seed_data()
        cls.settings = load_settings()
        cls.store = KnowledgeStore.from_settings(cls.settings)
        cls.tools = ToolRegistry.from_store(cls.store)

    def test_flight_search_returns_structured_payload(self) -> None:
        observation = self.tools.flight_search.run(
            origin_city_id="KHI",
            destination_city_id="DXB",
            departure_date="2026-05-10",
            traveler_count=1,
            max_stops=2,
        )
        self.assertEqual(observation.tool_name, "FlightSearchTool")
        self.assertIn("results", observation.payload)
        self.assertGreater(len(observation.payload["results"]), 0)

    def test_hotel_search_filters_by_city(self) -> None:
        observation = self.tools.hotel_search.run(
            city_id="DXB",
            check_in_date="2026-05-10",
            check_out_date="2026-05-13",
            traveler_count=1,
            hotel_min_rating=3.0,
        )
        self.assertTrue(all(row["city_id"] == "DXB" for row in observation.payload["results"]))


if __name__ == "__main__":
    unittest.main()
