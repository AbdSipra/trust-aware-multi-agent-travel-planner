from __future__ import annotations

from datetime import datetime, timezone

from src.state.schemas import ToolObservation
from src.tools.base import BaseTool


class CalendarConstraintTool(BaseTool):
    name = "CalendarConstraintTool"

    def run(self, **kwargs) -> ToolObservation:
        selected_flight = kwargs.get("selected_flight")
        selected_hotel = kwargs.get("selected_hotel")
        task = kwargs["task"]
        issues: list[str] = []
        if not selected_flight:
            issues.append("no_flight_selected")
        if not selected_hotel:
            issues.append("no_hotel_selected")
        if selected_flight:
            arrival_time = selected_flight["arrival_time"]
            if arrival_time > task.must_arrive_before:
                issues.append("arrival_after_deadline")
            if selected_flight["departure_time"] < task.must_depart_after:
                issues.append("departure_before_window")
        if selected_hotel:
            if selected_hotel["check_in_date"] > task.trip_start_date:
                issues.append("hotel_check_in_after_trip_start")
            if selected_hotel["check_out_date"] < task.trip_end_date:
                issues.append("hotel_check_out_before_trip_end")
        payload = {"valid": not issues, "issues": issues}
        return ToolObservation(
            observation_id=f"calendar-{task.task_id}",
            tool_name=self.name,
            payload=payload,
            source_name="deterministic_calendar_validator",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_version="1",
            freshness="fresh",
            verification_status="computed",
        )
