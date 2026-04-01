from __future__ import annotations

from dataclasses import dataclass

from src.tools.attraction_search import AttractionSearchTool
from src.tools.budget_calculator import BudgetCalculatorTool
from src.tools.calendar_constraint import CalendarConstraintTool
from src.tools.flight_search import FlightSearchTool
from src.tools.hotel_search import HotelSearchTool
from src.tools.knowledge_store import KnowledgeStore
from src.tools.route_time_estimator import RouteTimeEstimatorTool


@dataclass
class ToolRegistry:
    flight_search: FlightSearchTool
    hotel_search: HotelSearchTool
    attraction_search: AttractionSearchTool
    budget_calculator: BudgetCalculatorTool
    calendar_constraint: CalendarConstraintTool
    route_time_estimator: RouteTimeEstimatorTool

    @classmethod
    def from_store(cls, store: KnowledgeStore) -> "ToolRegistry":
        return cls(
            flight_search=FlightSearchTool(store),
            hotel_search=HotelSearchTool(store),
            attraction_search=AttractionSearchTool(store),
            budget_calculator=BudgetCalculatorTool(),
            calendar_constraint=CalendarConstraintTool(),
            route_time_estimator=RouteTimeEstimatorTool(store),
        )
