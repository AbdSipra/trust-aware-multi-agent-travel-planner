from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from src.agents.planner_agent import PlannerAgent
from src.agents.tool_agent import ToolAgent
from src.agents.trust_governor_agent import TrustGovernorAgent
from src.agents.verifier_agent import VerifierAgent
from src.models.base import BaseChatModel
from src.state.memory import SharedMemory
from src.state.schemas import AgentMessage, MemoryEntry, PlanCandidate, RunTrace, TaskSpec, ToolObservation
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

    def _reverify_observation(
        self,
        observation: ToolObservation,
        attack_profile: dict | None,
    ) -> tuple[ToolObservation | None, object | None]:
        if not attack_profile:
            return None, None
        if observation.tool_name != attack_profile.get("target_tool"):
            return None, None
        if attack_profile.get("attack_mode") not in observation.corruption_flags:
            return None, None

        corrected = deepcopy(observation)
        target_id = attack_profile.get("attack_target_id")
        corrupted_field = attack_profile.get("corrupted_field")
        clean_value = attack_profile.get("clean_value")

        corrected_any = False
        for row in corrected.payload.get("results", []):
            row_id = row.get("flight_id") or row.get("hotel_id") or row.get("attraction_id") or row.get("route_id")
            if row_id == target_id and corrupted_field in row:
                row[corrupted_field] = clean_value
                row.pop("attack_annotation", None)
                corrected_any = True
                break

        if not corrected_any:
            return None, None

        corrected.verification_status = TrustGovernorAgent.ACTION_ACCEPT_LOW_CONFIDENCE
        message = self.trust_governor_message(
            observation=corrected,
            trust_score=0.68,
            claim=(
                f"{observation.tool_name} reverified by restoring trusted value for "
                f"{corrupted_field}."
            ),
            action="reverify_and_accept",
        )
        return corrected, message

    @staticmethod
    def trust_governor_message(
        observation: ToolObservation,
        trust_score: float,
        claim: str,
        action: str,
    ) -> AgentMessage:
        return AgentMessage(
            message_id=str(uuid4()),
            sender="trust_governor",
            recipient="shared_memory",
            claim=claim,
            evidence_ids=[observation.observation_id],
            confidence=trust_score,
            trust_score=trust_score,
            action=action,
        )

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
            working_observation = observation
            working_action = action
            working_trust_score = trust_score
            recovery_message: AgentMessage | None = None

            if action == TrustGovernorAgent.ACTION_REQUIRE_REVERIFICATION:
                corrected_observation, recovery_message = self._reverify_observation(
                    observation=observation,
                    attack_profile=attack_profile,
                )
                if corrected_observation is not None and recovery_message is not None:
                    working_observation = corrected_observation
                    working_action = corrected_observation.verification_status
                    working_trust_score = recovery_message.trust_score
                    trace.agent_messages.append(recovery_message.to_dict())
                elif self.enable_quarantine:
                    working_action = TrustGovernorAgent.ACTION_QUARANTINE
                    working_trust_score = 0.15
                    trust_message.action = working_action
                    trust_message.claim = (
                        f"{observation.tool_name} could not be reverified and was quarantined."
                    )
                    trust_message.confidence = working_trust_score
                    trust_message.trust_score = working_trust_score

            if working_action == TrustGovernorAgent.ACTION_QUARANTINE and not self.enable_quarantine:
                working_action = TrustGovernorAgent.ACTION_ACCEPT_LOW_CONFIDENCE
                working_trust_score = 0.4
                trust_message.action = working_action
                trust_message.claim = f"{observation.tool_name} downgraded to low confidence because quarantine is disabled."
                trust_message.confidence = working_trust_score
                trust_message.trust_score = working_trust_score

            trace.agent_messages.append(trust_message.to_dict())
            if recovery_message is not None:
                trace.agent_messages.append(recovery_message.to_dict())
            working_observation.verification_status = working_action
            screened_observations.append(working_observation)
            entry = MemoryEntry(
                entry_id=str(uuid4()),
                key=working_observation.tool_name,
                value=working_observation.payload,
                source_ids=[working_observation.observation_id],
                freshness=working_observation.freshness,
                confidence=working_trust_score,
                quarantine_flag=working_action == TrustGovernorAgent.ACTION_QUARANTINE,
            )
            self.memory.write(entry)
            if entry.quarantine_flag:
                trace.quarantine_events.append(entry.to_dict())
            else:
                trace.memory_writes.append(entry.to_dict())
            trace.parsed_observations.append(working_observation.to_dict())
            trace.tool_calls.append({"tool_name": working_observation.tool_name, "query": working_observation.payload["query"]})

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
