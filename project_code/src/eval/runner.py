from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

from src.agents.langgraph_runtime import execute_with_langgraph, langgraph_available
from src.agents.naive_multi_agent import NaiveMultiAgentSharedMemorySystem
from src.agents.single_agent import SingleAgentToolUseSystem
from src.agents.trust_aware_multi_agent import TrustAwareMultiAgentSystem
from src.config.settings import Settings, load_settings
from src.eval.attacks import apply_attack_to_observation, select_attack_for_task
from src.eval.metrics import compute_run_metrics, summarize_metrics
from src.models.factory import build_chat_model
from src.state.schemas import RunTrace, TaskSpec
from src.tools.knowledge_store import KnowledgeStore
from src.tools.registry import ToolRegistry
from src.utils.io import read_jsonl


def _load_tasks(settings: Settings, task_split: str) -> list[TaskSpec]:
    records = read_jsonl(settings.tasks_dir / f"{task_split}.jsonl")
    return [TaskSpec(**record) for record in records]


def _load_attack_catalog(settings: Settings) -> list[dict]:
    return read_jsonl(settings.attacks_dir / "attack_catalog.jsonl")


def _build_system(system_variant: str, tools: ToolRegistry, settings: Settings):
    chat_model = build_chat_model(settings)
    if system_variant == "single_agent_tool_use":
        return SingleAgentToolUseSystem(tools, chat_model)
    if system_variant == "naive_multi_agent_shared_memory":
        return NaiveMultiAgentSharedMemorySystem(tools, chat_model)
    if system_variant == "trust_aware_multi_agent":
        return TrustAwareMultiAgentSystem(tools, chat_model)
    if system_variant == "ablation_no_quarantine":
        return TrustAwareMultiAgentSystem(tools, chat_model, enable_quarantine=False)
    if system_variant == "ablation_no_verifier":
        return TrustAwareMultiAgentSystem(tools, chat_model, enable_verifier=False)
    if system_variant == "ablation_no_provenance":
        return TrustAwareMultiAgentSystem(tools, chat_model, enable_provenance=False)
    raise ValueError(f"Unknown system_variant: {system_variant}")


def _attack_observations(observations: list, attack_profile: dict | None) -> list:
    return [apply_attack_to_observation(obs, attack_profile) for obs in observations]


def _patch_tools_with_attack(system, attack_profile: dict | None):
    originals = {}
    tool_names = ["flight_search", "hotel_search", "attraction_search"]
    for name in tool_names:
        tool = getattr(system.tools, name)
        originals[name] = tool.run

        def make_wrapper(original_run):
            def wrapper(**kwargs):
                observation = original_run(**kwargs)
                return apply_attack_to_observation(observation, attack_profile)

            return wrapper

        tool.run = make_wrapper(tool.run)
    return originals


def _restore_patched_tools(system, originals: dict) -> None:
    for name, original in originals.items():
        getattr(system.tools, name).run = original


def _execute_system(
    system_variant: str,
    system,
    task: TaskSpec,
    trace: RunTrace,
    attack_profile: dict | None,
    use_langgraph: bool,
):
    if use_langgraph and langgraph_available():
        return execute_with_langgraph(
            system_variant=system_variant,
            system=system,
            task=task,
            trace=trace,
            attack_profile=attack_profile,
        )
    if system_variant == "single_agent_tool_use":
        return system.run(task, trace)
    if system_variant == "naive_multi_agent_shared_memory":
        return system.run(task, trace)
    return system.run(task, trace, attack_profile=attack_profile)


def run_experiment(
    task_split: str,
    system_variant: str,
    attack_mode: str | None = None,
    seed: int | None = None,
    task_limit: int | None = None,
    settings: Settings | None = None,
    persist_traces: bool = True,
    runs_dir: Path | None = None,
    use_langgraph: bool = False,
) -> tuple[dict, list[RunTrace]]:
    settings = settings or load_settings()
    effective_runs_dir = runs_dir.resolve() if runs_dir is not None else settings.runs_dir
    if persist_traces:
        effective_runs_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(settings, task_split)
    if task_limit is not None:
        tasks = tasks[:task_limit]
    attack_catalog = _load_attack_catalog(settings)
    store = KnowledgeStore.from_settings(settings)
    tools = ToolRegistry.from_store(store)
    traces: list[RunTrace] = []

    for task in tasks:
        start = time.perf_counter()
        attack_profile = select_attack_for_task(task, attack_catalog, override_mode=attack_mode)
        system = _build_system(system_variant, tools, settings)
        trace = RunTrace(
            run_id=str(uuid4()),
            task_id=task.task_id,
            model_provider=settings.provider,
            model_name=settings.groq_model if settings.provider == "groq" else settings.ollama_model,
            system_variant=system_variant,
            attack_profile=attack_profile["attack_mode"] if attack_profile else None,
        )

        originals = _patch_tools_with_attack(system, attack_profile)
        try:
            candidate, trace = _execute_system(
                system_variant=system_variant,
                system=system,
                task=task,
                trace=trace,
                attack_profile=attack_profile,
                use_langgraph=use_langgraph,
            )
        finally:
            _restore_patched_tools(system, originals)

        trace.latency_ms = int((time.perf_counter() - start) * 1000)
        compute_run_metrics(trace, task=task, attack_profile=attack_profile)
        traces.append(trace)
        if persist_traces:
            output_path = effective_runs_dir / f"{trace.run_id}.json"
            output_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    return summarize_metrics(traces), traces
