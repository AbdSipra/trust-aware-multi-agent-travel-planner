from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_CODE_DIR))

from src.agents.planner_agent import PlannerAgent
from src.state.schemas import TaskSpec


class PlannerAgentTests(unittest.TestCase):
    def test_heuristic_candidate_prefers_budget_feasible_package(self) -> None:
        task = TaskSpec(
            task_id="T-1",
            user_query="Plan a trip",
            origin_city_id="KHI",
            destination_city_id="DXB",
            trip_start_date="2026-05-10",
            trip_end_date="2026-05-13",
            traveler_count=1,
            budget_limit_usd=700.0,
            must_visit_categories=["cultural", "family"],
            hotel_min_rating=3.0,
            max_stops=1,
            must_arrive_before="22:00",
            must_depart_after="06:00",
            hard_constraints={},
            soft_preferences={"prefer_city_center": True, "prefer_lower_cost": True},
            difficulty_level="medium",
        )
        flights = [
            {"flight_id": "F-1", "price_usd": "220.0", "stops": "0"},
            {"flight_id": "F-2", "price_usd": "340.0", "stops": "0"},
        ]
        hotels = [
            {"hotel_id": "H-1", "price_per_night_usd": "120.0", "distance_from_center_km": "0.4"},
            {"hotel_id": "H-2", "price_per_night_usd": "180.0", "distance_from_center_km": "0.2"},
        ]
        attractions = [
            {"attraction_id": "A-1", "category": "cultural", "ticket_price_usd": "20.0", "popularity_score": "0.95"},
            {"attraction_id": "A-2", "category": "family", "ticket_price_usd": "18.0", "popularity_score": "0.90"},
            {"attraction_id": "A-3", "category": "family", "ticket_price_usd": "60.0", "popularity_score": "0.99"},
        ]

        planner = PlannerAgent(chat_model=None)
        candidate = planner.select_candidate(task, flights, hotels, attractions)

        self.assertEqual(candidate["flight_id"], "F-1")
        self.assertEqual(candidate["hotel_id"], "H-1")
        self.assertEqual(set(candidate["attraction_ids"]), {"A-1", "A-2"})


if __name__ == "__main__":
    unittest.main()
