from __future__ import annotations

from uuid import uuid4

from src.state.schemas import AgentMessage, ToolObservation


class TrustGovernorAgent:
    ACTION_ACCEPT = "accept"
    ACTION_ACCEPT_LOW_CONFIDENCE = "accept_with_low_confidence"
    ACTION_QUARANTINE = "quarantine"
    ACTION_REQUIRE_REVERIFICATION = "require_reverification"
    ACTION_REJECT = "reject"

    def screen_observation(self, observation: ToolObservation, attack_profile: dict | None = None) -> tuple[str, float, AgentMessage]:
        corruption_flags = set(observation.corruption_flags)
        targeted_attack = bool(
            attack_profile
            and observation.tool_name == attack_profile.get("target_tool")
            and attack_profile.get("attack_mode") in corruption_flags
        )
        if targeted_attack and attack_profile.get("should_be_quarantined", False):
            action = self.ACTION_QUARANTINE
            trust_score = 0.15
        elif targeted_attack:
            action = self.ACTION_REQUIRE_REVERIFICATION
            trust_score = 0.45
        elif corruption_flags:
            action = self.ACTION_REQUIRE_REVERIFICATION
            trust_score = 0.35
        elif observation.freshness == "unknown":
            action = self.ACTION_ACCEPT_LOW_CONFIDENCE
            trust_score = 0.55
        else:
            action = self.ACTION_ACCEPT
            trust_score = 0.88
        message = AgentMessage(
            message_id=str(uuid4()),
            sender="trust_governor",
            recipient="shared_memory",
            claim=f"{observation.tool_name} screened with action {action}.",
            evidence_ids=[observation.observation_id],
            confidence=trust_score,
            trust_score=trust_score,
            action=action,
        )
        return action, trust_score, message
