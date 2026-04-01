from __future__ import annotations

from datetime import datetime, timezone

from src.state.schemas import ToolObservation
from src.tools.base import BaseTool
from src.tools.knowledge_store import KnowledgeStore


class FlightSearchTool(BaseTool):
    name = "FlightSearchTool"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def run(self, **kwargs) -> ToolObservation:
        origin_city_id = kwargs["origin_city_id"]
        destination_city_id = kwargs["destination_city_id"]
        departure_date = kwargs["departure_date"]
        max_stops = int(kwargs.get("max_stops", 2))
        limit = int(kwargs.get("limit", 8))

        matches = [
            row
            for row in self.store.flights
            if row["origin_city_id"] == origin_city_id
            and row["destination_city_id"] == destination_city_id
            and row["departure_date"] == departure_date
            and int(row["stops"]) <= max_stops
            and int(row["seats_available"]) >= int(kwargs.get("traveler_count", 1))
        ]
        matches.sort(key=lambda row: (float(row["price_usd"]), int(row["stops"])))
        payload = {
            "query": {
                "origin_city_id": origin_city_id,
                "destination_city_id": destination_city_id,
                "departure_date": departure_date,
                "max_stops": max_stops,
            },
            "results": matches[:limit],
            "result_count": len(matches),
        }
        record_version = matches[0]["record_version"] if matches else "none"
        freshness = "fresh" if matches else "unknown"
        return ToolObservation(
            observation_id=f"flight-{origin_city_id}-{destination_city_id}-{departure_date}",
            tool_name=self.name,
            payload=payload,
            source_name="local_flights_csv",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_version=record_version,
            freshness=freshness,
            verification_status="raw",
        )
