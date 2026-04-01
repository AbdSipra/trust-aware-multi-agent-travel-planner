from __future__ import annotations

from datetime import datetime, timezone

from src.state.schemas import ToolObservation
from src.tools.base import BaseTool
from src.tools.knowledge_store import KnowledgeStore


class RouteTimeEstimatorTool(BaseTool):
    name = "RouteTimeEstimatorTool"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def run(self, **kwargs) -> ToolObservation:
        city_id = kwargs["city_id"]
        from_node_type = kwargs["from_node_type"]
        from_node_id = kwargs["from_node_id"]
        to_node_type = kwargs["to_node_type"]
        to_node_id = kwargs["to_node_id"]

        matches = [
            row
            for row in self.store.routes
            if row["city_id"] == city_id
            and row["from_node_type"] == from_node_type
            and row["from_node_id"] == from_node_id
            and row["to_node_type"] == to_node_type
            and row["to_node_id"] == to_node_id
        ]
        payload = {
            "query": {
                "city_id": city_id,
                "from_node_type": from_node_type,
                "from_node_id": from_node_id,
                "to_node_type": to_node_type,
                "to_node_id": to_node_id,
            },
            "results": matches[:3],
        }
        record_version = matches[0]["last_updated"] if matches else "none"
        return ToolObservation(
            observation_id=f"route-{city_id}-{from_node_id}-{to_node_id}",
            tool_name=self.name,
            payload=payload,
            source_name="local_routes_csv",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_version=record_version,
            freshness="fresh" if matches else "unknown",
            verification_status="raw",
        )
