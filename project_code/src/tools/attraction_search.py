from __future__ import annotations

from datetime import datetime, timezone

from src.state.schemas import ToolObservation
from src.tools.base import BaseTool
from src.tools.knowledge_store import KnowledgeStore


class AttractionSearchTool(BaseTool):
    name = "AttractionSearchTool"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def run(self, **kwargs) -> ToolObservation:
        city_id = kwargs["city_id"]
        categories = set(kwargs.get("categories", []))
        limit = int(kwargs.get("limit", 12))

        matches = [
            row
            for row in self.store.attractions
            if row["city_id"] == city_id and (not categories or row["category"] in categories)
        ]
        matches.sort(
            key=lambda row: (
                float(row["ticket_price_usd"]),
                -float(row["popularity_score"]),
            )
        )
        payload = {
            "query": {"city_id": city_id, "categories": sorted(categories)},
            "results": matches[:limit],
            "result_count": len(matches),
        }
        record_version = matches[0]["record_version"] if matches else "none"
        return ToolObservation(
            observation_id=f"attraction-{city_id}-{'-'.join(sorted(categories)) or 'all'}",
            tool_name=self.name,
            payload=payload,
            source_name="local_attractions_csv",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_version=record_version,
            freshness="fresh" if matches else "unknown",
            verification_status="raw",
        )
