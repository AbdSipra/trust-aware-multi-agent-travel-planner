from __future__ import annotations

from datetime import datetime, timezone

from src.state.schemas import ToolObservation
from src.tools.base import BaseTool
from src.tools.knowledge_store import KnowledgeStore


class HotelSearchTool(BaseTool):
    name = "HotelSearchTool"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def run(self, **kwargs) -> ToolObservation:
        city_id = kwargs["city_id"]
        check_in = kwargs["check_in_date"]
        check_out = kwargs["check_out_date"]
        min_rating = float(kwargs.get("hotel_min_rating", 0))
        traveler_count = int(kwargs.get("traveler_count", 1))
        limit = int(kwargs.get("limit", 8))

        matches = [
            row
            for row in self.store.hotels
            if row["city_id"] == city_id
            and float(row["star_rating"]) >= min_rating
            and row["check_in_date"] <= check_in
            and row["check_out_date"] >= check_out
            and int(row["rooms_available"]) >= max(1, traveler_count)
        ]
        matches.sort(
            key=lambda row: (
                float(row["price_per_night_usd"]),
                float(row["distance_from_center_km"]),
                -float(row["star_rating"]),
            )
        )
        payload = {
            "query": {
                "city_id": city_id,
                "check_in_date": check_in,
                "check_out_date": check_out,
                "hotel_min_rating": min_rating,
            },
            "results": matches[:limit],
            "result_count": len(matches),
        }
        record_version = matches[0]["record_version"] if matches else "none"
        return ToolObservation(
            observation_id=f"hotel-{city_id}-{check_in}-{check_out}",
            tool_name=self.name,
            payload=payload,
            source_name="local_hotels_csv",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_version=record_version,
            freshness="fresh" if matches else "unknown",
            verification_status="raw",
        )
