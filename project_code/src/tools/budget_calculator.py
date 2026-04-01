from __future__ import annotations

from datetime import datetime, timezone

from src.state.schemas import ToolObservation
from src.tools.base import BaseTool


class BudgetCalculatorTool(BaseTool):
    name = "BudgetCalculatorTool"

    def run(self, **kwargs) -> ToolObservation:
        selected_flight = kwargs.get("selected_flight")
        selected_hotel = kwargs.get("selected_hotel")
        selected_attractions = kwargs.get("selected_attractions", [])
        traveler_count = int(kwargs.get("traveler_count", 1))
        stay_nights = int(kwargs.get("stay_nights", 1))

        flight_cost = float(selected_flight["price_usd"]) * traveler_count if selected_flight else 0.0
        hotel_cost = (
            float(selected_hotel["price_per_night_usd"]) * stay_nights * max(1, traveler_count // 2)
            if selected_hotel
            else 0.0
        )
        attraction_cost = sum(float(item["ticket_price_usd"]) for item in selected_attractions) * traveler_count
        total_cost = round(flight_cost + hotel_cost + attraction_cost, 2)
        payload = {
            "flight_cost_usd": round(flight_cost, 2),
            "hotel_cost_usd": round(hotel_cost, 2),
            "attraction_cost_usd": round(attraction_cost, 2),
            "total_cost_usd": total_cost,
        }
        return ToolObservation(
            observation_id=f"budget-{selected_flight['flight_id'] if selected_flight else 'none'}",
            tool_name=self.name,
            payload=payload,
            source_name="computed_budget",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_version="1",
            freshness="fresh",
            verification_status="computed",
        )
