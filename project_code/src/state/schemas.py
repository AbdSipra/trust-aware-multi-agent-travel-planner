from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    task_id: str
    user_query: str
    origin_city_id: str
    destination_city_id: str
    trip_start_date: str
    trip_end_date: str
    traveler_count: int
    budget_limit_usd: float
    must_visit_categories: list[str]
    hotel_min_rating: float
    max_stops: int
    must_arrive_before: str
    must_depart_after: str
    hard_constraints: dict[str, Any]
    soft_preferences: dict[str, Any]
    difficulty_level: str
    expected_attack_profile: str | None = None
    notes_for_human_eval: str = ""
    attack_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    observation_id: str
    tool_name: str
    payload: dict[str, Any]
    source_name: str
    timestamp: str
    record_version: str
    freshness: str
    verification_status: str
    corruption_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentMessage:
    message_id: str
    sender: str
    recipient: str
    claim: str
    evidence_ids: list[str]
    confidence: float
    trust_score: float
    action: str = "inform"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEntry:
    entry_id: str
    key: str
    value: dict[str, Any]
    source_ids: list[str]
    freshness: str
    confidence: float
    quarantine_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanCandidate:
    plan_id: str
    itinerary_steps: list[dict[str, Any]]
    total_estimated_cost_usd: float
    constraint_status: str
    unresolved_issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunTrace:
    run_id: str
    task_id: str
    model_provider: str
    model_name: str
    system_variant: str
    attack_profile: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    parsed_observations: list[dict[str, Any]] = field(default_factory=list)
    agent_messages: list[dict[str, Any]] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    quarantine_events: list[dict[str, Any]] = field(default_factory=list)
    verifier_decisions: list[dict[str, Any]] = field(default_factory=list)
    final_itinerary: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
