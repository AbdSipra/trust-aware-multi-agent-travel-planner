from __future__ import annotations

from uuid import uuid4

from src.agents.planner_agent import PlannerAgent
from src.agents.tool_agent import ToolAgent
from src.agents.trust_governor_agent import TrustGovernorAgent
from src.agents.verifier_agent import VerifierAgent
from src.models.base import BaseChatModel
from src.state.memory import SharedMemory
from src.state.schemas import MemoryEntry, PlanCandidate, RunTrace, TaskSpec, ToolObservation
from src.tools.registry import ToolRegistry


class TrustAwareMultiAgentSystem:
    def __init__(
        self,
        tools: ToolRegistry,
        chat_model: BaseChatModel | None = None,
        enable_quarantine: bool = True,
        enable_verifier: bool = True,
        enable_provenance: bool = True,
    ) -> None:
        self.tools = tools
        self.planner = PlannerAgent(chat_model=chat_model)
        self.tool_agent = ToolAgent(tools=tools)
        self.verifier = VerifierAgent(tools=tools)
        self.trust_governor = TrustGovernorAgent()
        self.memory = SharedMemory()
        self.enable_quarantine = enable_quarantine
        self.enable_verifier = enable_verifier
        self.enable_provenance = enable_provenance

    def run(
        self,
        task: TaskSpec,
        trace: RunTrace,
        attack_profile: dict | None = None,
    ) -> tuple[PlanCandidate | None, RunTrace]:
        search_plan, planner_message = self.planner.plan_search(task)
        trace.agent_messages.append(planner_message.to_dict())

        observations, tool_messages = self.tool_agent.execute_search_plan(search_plan)
        trace.agent_messages.extend(message.to_dict() for message in tool_messages)

        screened_observations: list[ToolObservation] = []
        for observation in observations:
            if not self.enable_provenance:
                observation.corruption_flags = []
            action, trust_score, trust_message = self.trust_governor.screen_observation(
                observation=observation,
                attack_profile=attack_profile,
            )
            if action == TrustGovernorAgent.ACTION_QUARANTINE and not self.enable_quarantine:
                action = TrustGovernorAgent.ACTION_ACCEPT_LOW_CONFIDENCE
                trust_score = 0.4
                trust_message.action = action
                trust_message.claim = f"{observation.tool_name} downgraded to low confidence because quarantine is disabled."
            trace.agent_messages.append(trust_message.to_dict())
            observation.verification_status = action
            screened_observations.append(observation)
            entry = MemoryEntry(
                entry_id=str(uuid4()),
                key=observation.tool_name,
                value=observation.payload,
                source_ids=[observation.observation_id],
                freshness=observation.freshness,
                confidence=trust_score,
                quarantine_flag=action == TrustGovernorAgent.ACTION_QUARANTINE,
            )
            self.memory.write(entry)
            if entry.quarantine_flag:
                trace.quarantine_events.append(entry.to_dict())
            else:
                trace.memory_writes.append(entry.to_dict())
            trace.parsed_observations.append(observation.to_dict())
            trace.tool_calls.append({"tool_name": observation.tool_name, "query": observation.payload["query"]})

        accepted_results = {
            observation.tool_name: observation.payload["results"]
            for observation in screened_observations
            if observation.verification_status != TrustGovernorAgent.ACTION_QUARANTINE
        }

        flights = accepted_results.get("FlightSearchTool", [])
        hotels = accepted_results.get("HotelSearchTool", [])
        attractions = accepted_results.get("AttractionSearchTool", [])
        if not flights or not hotels:
            trace.failure_reason = "critical_information_quarantined"
            return None, trace

        selection = self.planner.select_candidate(task=task, flights=flights, hotels=hotels, attractions=attractions)
        selected_flight = next((item for item in flights if item["flight_id"] == selection.get("flight_id")), flights[0])
        selected_hotel = next((item for item in hotels if item["hotel_id"] == selection.get("hotel_id")), hotels[0])
        attraction_ids = set(selection.get("attraction_ids", []))
        selected_attractions = [item for item in attractions if item["attraction_id"] in attraction_ids] or attractions[:3]

        if not self.enable_verifier:
            candidate = PlanCandidate(
                plan_id=str(uuid4()),
                itinerary_steps=[
                    {"type": "flight", "selection": selected_flight},
                    {"type": "hotel", "selection": selected_hotel},
                    {"type": "attractions", "selection": selected_attractions},
                ],
                total_estimated_cost_usd=0.0,
                constraint_status="unverified",
                unresolved_issues=["verifier_disabled"],
            )
            trace.final_itinerary = candidate.to_dict()
            return candidate, trace

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
