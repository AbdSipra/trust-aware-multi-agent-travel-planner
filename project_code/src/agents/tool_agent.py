from __future__ import annotations

from uuid import uuid4

from src.state.schemas import AgentMessage, ToolObservation
from src.tools.registry import ToolRegistry


class ToolAgent:
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    def execute_search_plan(self, search_plan: dict) -> tuple[list[ToolObservation], list[AgentMessage]]:
        observations = [
            self.tools.flight_search.run(**search_plan["flight_query"]),
            self.tools.hotel_search.run(**search_plan["hotel_query"]),
            self.tools.attraction_search.run(**search_plan["attraction_query"]),
        ]
        messages = [
            AgentMessage(
                message_id=str(uuid4()),
                sender="tool_agent",
                recipient="planner",
                claim=f"{obs.tool_name} returned {obs.payload.get('result_count', len(obs.payload.get('results', [])))} results.",
                evidence_ids=[obs.observation_id],
                confidence=0.74,
                trust_score=0.74,
                action="report_tool_results",
            )
            for obs in observations
        ]
        return observations, messages
