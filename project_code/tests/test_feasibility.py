from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_CODE_DIR))

from src.eval.feasibility import rebalance_tasks_to_feasibility


class FeasibilityTests(unittest.TestCase):
    def test_rebalance_raises_infeasible_budget_and_updates_query(self) -> None:
        tasks = [
            {
                "task_id": "EVAL-01",
                "user_query": "The budget for this trip is set at $500. Please include cultural and family activities.",
                "origin_city_id": "KHI",
                "destination_city_id": "DXB",
                "trip_start_date": "2026-05-10",
                "trip_end_date": "2026-05-13",
                "traveler_count": 2,
                "budget_limit_usd": 500.0,
                "must_visit_categories": ["cultural", "family"],
                "hotel_min_rating": 3.0,
                "max_stops": 1,
                "must_arrive_before": "22:00",
                "must_depart_after": "06:00",
                "hard_constraints": {"require_breakfast": True, "avoid_red_eye": True},
                "soft_preferences": {"prefer_city_center": True, "prefer_lower_cost": True},
                "difficulty_level": "medium",
                "expected_attack_profile": None,
                "notes_for_human_eval": "",
                "attack_id": None,
            }
        ]
        flights = [
            {
                "flight_id": "FL-KHI-DXB-0510",
                "origin_city_id": "KHI",
                "destination_city_id": "DXB",
                "departure_date": "2026-05-10",
                "departure_time": "10:00",
                "arrival_time": "12:00",
                "stops": "0",
                "price_usd": "180.00",
                "seats_available": "4",
                "baggage_included": "yes",
            }
        ]
        hotels = [
            {
                "hotel_id": "HT-DXB-01",
                "city_id": "DXB",
                "star_rating": "4.0",
                "price_per_night_usd": "120.00",
                "check_in_date": "2026-05-01",
                "check_out_date": "2026-08-31",
                "rooms_available": "3",
                "distance_from_center_km": "1.0",
                "breakfast_included": "yes",
            }
        ]
        attractions = [
            {
                "attraction_id": "AT-DXB-01",
                "city_id": "DXB",
                "category": "cultural",
                "ticket_price_usd": "18.00",
                "popularity_score": "0.9",
            },
            {
                "attraction_id": "AT-DXB-02",
                "city_id": "DXB",
                "category": "family",
                "ticket_price_usd": "22.00",
                "popularity_score": "0.8",
            },
        ]

        adjusted, audit_rows = rebalance_tasks_to_feasibility(tasks, flights, hotels, attractions)

        self.assertEqual(audit_rows[0]["status"], "budget_raised")
        self.assertGreater(adjusted[0]["budget_limit_usd"], 500.0)
        self.assertIn(f"${int(adjusted[0]['budget_limit_usd'])}", adjusted[0]["user_query"])


if __name__ == "__main__":
    unittest.main()
