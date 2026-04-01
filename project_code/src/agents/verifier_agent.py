from __future__ import annotations

from uuid import uuid4

from src.state.schemas import AgentMessage, TaskSpec
from src.tools.registry import ToolRegistry


class VerifierAgent:
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    def verify(
        self,
        task: TaskSpec,
        selected_flight: dict | None,
        selected_hotel: dict | None,
        selected_attractions: list[dict],
    ) -> tuple[dict, AgentMessage]:
        stay_nights = max(1, (int(task.trip_end_date[-2:]) - int(task.trip_start_date[-2:])))
        budget_obs = self.tools.budget_calculator.run(
            selected_flight=selected_flight,
            selected_hotel=selected_hotel,
            selected_attractions=selected_attractions,
            traveler_count=task.traveler_count,
            stay_nights=stay_nights,
        )
        calendar_obs = self.tools.calendar_constraint.run(
            selected_flight=selected_flight,
            selected_hotel=selected_hotel,
            task=task,
        )
        issues = list(calendar_obs.payload["issues"])
        if selected_flight and int(selected_flight.get("seats_available", "0")) < task.traveler_count:
            issues.append("flight_unavailable")
        if selected_hotel and int(selected_hotel.get("rooms_available", "0")) < max(1, task.traveler_count):
            issues.append("hotel_unavailable")
        if selected_flight and not selected_flight.get("baggage_included"):
            issues.append("missing_flight_field")
        if budget_obs.payload["total_cost_usd"] > task.budget_limit_usd:
            issues.append("budget_exceeded")
        verifier_payload = {
            "valid": not issues,
            "issues": issues,
            "budget": budget_obs.payload,
            "calendar": calendar_obs.payload,
        }
        message = AgentMessage(
            message_id=str(uuid4()),
            sender="verifier",
            recipient="planner",
            claim="Verification completed.",
            evidence_ids=[budget_obs.observation_id, calendar_obs.observation_id],
            confidence=0.9 if not issues else 0.65,
            trust_score=0.9 if not issues else 0.65,
            action="verification_result",
        )
        return verifier_payload, message
