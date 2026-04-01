from __future__ import annotations

import json
from uuid import uuid4

from src.models.base import BaseChatModel
from src.state.schemas import AgentMessage, TaskSpec


class PlannerAgent:
    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self.chat_model = chat_model

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
                return parsed
            except Exception:
                pass
        selected_flight = flights[0] if flights else None
        selected_hotel = hotels[0] if hotels else None
        selected_attractions = attractions[: min(3, len(attractions))]
        return {
            "flight_id": selected_flight["flight_id"] if selected_flight else None,
            "hotel_id": selected_hotel["hotel_id"] if selected_hotel else None,
            "attraction_ids": [item["attraction_id"] for item in selected_attractions],
        }
