from __future__ import annotations

from datetime import date
from itertools import combinations
import json
from uuid import uuid4

from src.models.base import BaseChatModel
from src.state.schemas import AgentMessage, TaskSpec


class PlannerAgent:
    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self.chat_model = chat_model

    @staticmethod
    def _stay_nights(task: TaskSpec) -> int:
        start = date.fromisoformat(task.trip_start_date)
        end = date.fromisoformat(task.trip_end_date)
        return max(1, (end - start).days)

    def _estimate_total_cost(
        self,
        task: TaskSpec,
        flight: dict | None,
        hotel: dict | None,
        attractions: list[dict],
    ) -> float:
        stay_nights = self._stay_nights(task)
        flight_cost = float(flight.get("price_usd", 0.0)) * task.traveler_count if flight else 0.0
        hotel_cost = float(hotel.get("price_per_night_usd", 0.0)) * stay_nights if hotel else 0.0
        attraction_cost = sum(float(item.get("ticket_price_usd", 0.0)) for item in attractions) * task.traveler_count
        return round(flight_cost + hotel_cost + attraction_cost, 2)

    @staticmethod
    def _selection_valid(
        parsed: dict,
        flights: list[dict],
        hotels: list[dict],
        attractions: list[dict],
    ) -> bool:
        flight_ids = {item["flight_id"] for item in flights}
        hotel_ids = {item["hotel_id"] for item in hotels}
        attraction_ids = {item["attraction_id"] for item in attractions}
        selected_attractions = parsed.get("attraction_ids", []) or []
        return (
            parsed.get("flight_id") in flight_ids
            and parsed.get("hotel_id") in hotel_ids
            and all(item in attraction_ids for item in selected_attractions)
        )

    def _choose_attraction_bundle(self, task: TaskSpec, attractions: list[dict]) -> list[dict]:
        if not attractions:
            return []

        pool = attractions[: min(6, len(attractions))]
        requested_categories = set(task.must_visit_categories)
        bundle_candidates: list[tuple[tuple[int, int, float, float], list[dict]]] = []

        for bundle_size in range(1, min(3, len(pool)) + 1):
            for bundle in combinations(pool, bundle_size):
                bundle_list = list(bundle)
                covered_categories = {item["category"] for item in bundle_list}
                coverage = len(covered_categories & requested_categories)
                popularity = sum(float(item.get("popularity_score", 0.0)) for item in bundle_list)
                cost = sum(float(item.get("ticket_price_usd", 0.0)) for item in bundle_list) * task.traveler_count
                score = (
                    coverage,
                    -int(cost * 100),
                    len(bundle_list) * -1,
                    int(popularity * 100),
                )
                bundle_candidates.append((score, bundle_list))

        if not bundle_candidates:
            return pool[: min(3, len(pool))]

        bundle_candidates.sort(key=lambda item: item[0], reverse=True)
        return bundle_candidates[0][1]

    def _heuristic_candidate(
        self,
        task: TaskSpec,
        flights: list[dict],
        hotels: list[dict],
        attractions: list[dict],
    ) -> dict:
        if not flights or not hotels:
            return {"flight_id": None, "hotel_id": None, "attraction_ids": []}

        attraction_bundle = self._choose_attraction_bundle(task, attractions)
        best_choice: tuple[tuple[int, int, int, int], dict] | None = None

        for flight in flights[: min(5, len(flights))]:
            for hotel in hotels[: min(8, len(hotels))]:
                total_cost = self._estimate_total_cost(task, flight, hotel, attraction_bundle)
                within_budget = total_cost <= task.budget_limit_usd
                distance_score = -int(float(hotel.get("distance_from_center_km", 99.0)) * 100)
                stops_score = -int(flight.get("stops", 9))
                cost_score = -int(total_cost * 100)
                coverage_score = len({item["category"] for item in attraction_bundle} & set(task.must_visit_categories))
                score = (
                    1 if within_budget else 0,
                    coverage_score,
                    cost_score,
                    distance_score if task.soft_preferences.get("prefer_city_center") else 0,
                    stops_score,
                )
                candidate = {
                    "flight_id": flight["flight_id"],
                    "hotel_id": hotel["hotel_id"],
                    "attraction_ids": [item["attraction_id"] for item in attraction_bundle],
                }
                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, candidate)

        return best_choice[1] if best_choice else {"flight_id": flights[0]["flight_id"], "hotel_id": hotels[0]["hotel_id"], "attraction_ids": [item["attraction_id"] for item in attraction_bundle]}

    def plan_search(self, task: TaskSpec) -> tuple[dict, AgentMessage]:
        search_plan = {
            "flight_query": {
                "origin_city_id": task.origin_city_id,
                "destination_city_id": task.destination_city_id,
                "departure_date": task.trip_start_date,
                "traveler_count": task.traveler_count,
                "max_stops": task.max_stops,
            },
            "hotel_query": {
                "city_id": task.destination_city_id,
                "check_in_date": task.trip_start_date,
                "check_out_date": task.trip_end_date,
                "traveler_count": task.traveler_count,
                "hotel_min_rating": task.hotel_min_rating,
            },
            "attraction_query": {
                "city_id": task.destination_city_id,
                "categories": task.must_visit_categories,
            },
        }
        if self.chat_model:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a planning agent. Return a JSON object with keys "
                        "flight_query, hotel_query, and attraction_query using the task fields."
                    ),
                },
                {"role": "user", "content": json.dumps(task.to_dict())},
            ]
            try:
                response = self.chat_model.generate(prompt, temperature=0.0, max_tokens=500, json_mode=True)
                parsed = json.loads(response.content)
                for key in ("flight_query", "hotel_query", "attraction_query"):
                    if key in parsed:
                        search_plan[key].update(parsed[key])
            except Exception:
                pass
        message = AgentMessage(
            message_id=str(uuid4()),
            sender="planner",
            recipient="tool_agent",
            claim="Search plan prepared for flight, hotel, and attraction tools.",
            evidence_ids=[],
            confidence=0.82,
            trust_score=0.82,
            action="request_tool_calls",
        )
        return search_plan, message

    def select_candidate(
        self,
        task: TaskSpec,
        flights: list[dict],
        hotels: list[dict],
        attractions: list[dict],
    ) -> dict:
        heuristic_candidate = self._heuristic_candidate(task, flights, hotels, attractions)
        if self.chat_model:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Select the best travel package and return JSON with keys "
                        "flight_id, hotel_id, attraction_ids. Prefer cheaper valid plans."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task.to_dict(),
                            "flights": flights[:5],
                            "hotels": hotels[:5],
                            "attractions": attractions[:8],
                        }
                    ),
                },
            ]
            try:
                response = self.chat_model.generate(prompt, temperature=0.1, max_tokens=600, json_mode=True)
                parsed = json.loads(response.content)
                if self._selection_valid(parsed, flights, hotels, attractions):
                    selected_flight = next(item for item in flights if item["flight_id"] == parsed["flight_id"])
                    selected_hotel = next(item for item in hotels if item["hotel_id"] == parsed["hotel_id"])
                    selected_attractions = [
                        item for item in attractions if item["attraction_id"] in set(parsed.get("attraction_ids", []))
                    ]
                    parsed_cost = self._estimate_total_cost(task, selected_flight, selected_hotel, selected_attractions)

                    heuristic_flight = next((item for item in flights if item["flight_id"] == heuristic_candidate["flight_id"]), None)
                    heuristic_hotel = next((item for item in hotels if item["hotel_id"] == heuristic_candidate["hotel_id"]), None)
                    heuristic_attractions = [
                        item for item in attractions if item["attraction_id"] in set(heuristic_candidate["attraction_ids"])
                    ]
                    heuristic_cost = self._estimate_total_cost(task, heuristic_flight, heuristic_hotel, heuristic_attractions)

                    if parsed_cost <= task.budget_limit_usd or heuristic_cost > task.budget_limit_usd:
                        return parsed
            except Exception:
                pass
        return heuristic_candidate
