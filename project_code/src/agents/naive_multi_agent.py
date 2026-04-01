from __future__ import annotations

from uuid import uuid4

from src.agents.planner_agent import PlannerAgent
from src.agents.tool_agent import ToolAgent
from src.agents.verifier_agent import VerifierAgent
from src.models.base import BaseChatModel
from src.state.memory import SharedMemory
from src.state.schemas import MemoryEntry, PlanCandidate, RunTrace, TaskSpec
from src.tools.registry import ToolRegistry


class NaiveMultiAgentSharedMemorySystem:
    def __init__(self, tools: ToolRegistry, chat_model: BaseChatModel | None = None) -> None:
        self.tools = tools
        self.planner = PlannerAgent(chat_model=chat_model)
        self.tool_agent = ToolAgent(tools=tools)
        self.verifier = VerifierAgent(tools=tools)
        self.memory = SharedMemory()

    def run(self, task: TaskSpec, trace: RunTrace) -> tuple[PlanCandidate | None, RunTrace]:
        search_plan, planner_message = self.planner.plan_search(task)
        trace.agent_messages.append(planner_message.to_dict())

        observations, tool_messages = self.tool_agent.execute_search_plan(search_plan)
        trace.agent_messages.extend(message.to_dict() for message in tool_messages)
        trace.parsed_observations.extend(obs.to_dict() for obs in observations)
        trace.tool_calls.extend(
            {"tool_name": obs.tool_name, "query": obs.payload["query"]} for obs in observations
        )

        for observation in observations:
            entry = MemoryEntry(
                entry_id=str(uuid4()),
                key=observation.tool_name,
                value=observation.payload,
                source_ids=[observation.observation_id],
                freshness=observation.freshness,
                confidence=0.8,
                quarantine_flag=False,
            )
            self.memory.write(entry)
            trace.memory_writes.append(entry.to_dict())

        flight_obs, hotel_obs, attraction_obs = observations
        selection = self.planner.select_candidate(
            task=task,
            flights=flight_obs.payload["results"],
            hotels=hotel_obs.payload["results"],
            attractions=attraction_obs.payload["results"],
        )
        selected_flight = next(
            (item for item in flight_obs.payload["results"] if item["flight_id"] == selection.get("flight_id")),
            flight_obs.payload["results"][0] if flight_obs.payload["results"] else None,
        )
        selected_hotel = next(
            (item for item in hotel_obs.payload["results"] if item["hotel_id"] == selection.get("hotel_id")),
            hotel_obs.payload["results"][0] if hotel_obs.payload["results"] else None,
        )
        attraction_ids = set(selection.get("attraction_ids", []))
        selected_attractions = [
            item for item in attraction_obs.payload["results"] if item["attraction_id"] in attraction_ids
        ] or attraction_obs.payload["results"][:3]

        verifier_payload, verifier_message = self.verifier.verify(
            task=task,
            selected_flight=selected_flight,
            selected_hotel=selected_hotel,
            selected_attractions=selected_attractions,
        )
        trace.verifier_decisions.append(verifier_payload)
        trace.agent_messages.append(verifier_message.to_dict())
        if not verifier_payload["valid"]:
            trace.failure_reason = ";".join(verifier_payload["issues"])
            return None, trace

        candidate = PlanCandidate(
            plan_id=str(uuid4()),
            itinerary_steps=[
                {"type": "flight", "selection": selected_flight},
                {"type": "hotel", "selection": selected_hotel},
                {"type": "attractions", "selection": selected_attractions},
            ],
            total_estimated_cost_usd=verifier_payload["budget"]["total_cost_usd"],
            constraint_status="valid",
            unresolved_issues=[],
        )
        trace.final_itinerary = candidate.to_dict()
        return candidate, trace
