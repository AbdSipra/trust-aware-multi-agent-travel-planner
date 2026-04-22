from __future__ import annotations

from typing import TypedDict

from src.agents.naive_multi_agent import NaiveMultiAgentSharedMemorySystem
from src.agents.single_agent import SingleAgentToolUseSystem
from src.agents.trust_aware_multi_agent import TrustAwareMultiAgentSystem
from src.state.schemas import RunTrace, TaskSpec


class PlannerGraphState(TypedDict):
    task: TaskSpec
    trace: RunTrace
    attack_profile: dict | None
    result: dict


def build_graph(system_variant: str, system):
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("LangGraph is not installed. Install dependencies first.") from exc

    workflow = StateGraph(PlannerGraphState)

    def execute(state: PlannerGraphState) -> PlannerGraphState:
        task = state["task"]
        trace = state["trace"]
        attack_profile = state.get("attack_profile")
        if isinstance(system, SingleAgentToolUseSystem):
            candidate, trace = system.run(task, trace)
        elif isinstance(system, NaiveMultiAgentSharedMemorySystem):
            candidate, trace = system.run(task, trace)
        elif isinstance(system, TrustAwareMultiAgentSystem):
            candidate, trace = system.run(task, trace, attack_profile=attack_profile)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported system for graph execution: {system_variant}")
        state["trace"] = trace
        state["result"] = candidate.to_dict() if candidate else {}
        return state

    workflow.add_node("execute", execute)
    workflow.add_edge(START, "execute")
    workflow.add_edge("execute", END)
    return workflow.compile()
