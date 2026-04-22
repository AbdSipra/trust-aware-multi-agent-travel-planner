from __future__ import annotations

from src.agents.graph_builder import build_graph
from src.state.schemas import PlanCandidate, RunTrace, TaskSpec


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401

        return True
    except Exception:
        return False


def execute_with_langgraph(
    system_variant: str,
    system,
    task: TaskSpec,
    trace: RunTrace,
    attack_profile: dict | None = None,
) -> tuple[PlanCandidate | None, RunTrace]:
    if not langgraph_available():
        raise RuntimeError("LangGraph is not installed. Install dependencies first.")

    graph = build_graph(system_variant=system_variant, system=system)
    final_state = graph.invoke(
        {
            "task": task,
            "trace": trace,
            "attack_profile": attack_profile,
            "result": {},
        }
    )
    candidate_payload = final_state.get("result", {})
    candidate = PlanCandidate(**candidate_payload) if candidate_payload else None
    return candidate, final_state["trace"]
