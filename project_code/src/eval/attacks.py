from __future__ import annotations

from copy import deepcopy

from src.state.schemas import TaskSpec, ToolObservation


def apply_attack_to_observation(observation: ToolObservation, attack_profile: dict | None) -> ToolObservation:
    if not attack_profile:
        return observation
    attacked = deepcopy(observation)
    target_tool = attack_profile["target_tool"]
    attack_mode = attack_profile["attack_mode"]
    if attacked.tool_name != target_tool:
        return attacked
    attacked.corruption_flags.append(attack_mode)
    corrupted_field = attack_profile["corrupted_field"]
    corrupted_value = attack_profile["corrupted_value"]
    target_id = attack_profile["attack_target_id"]
    for row in attacked.payload.get("results", []):
        row_id = (
            row.get("flight_id")
            or row.get("hotel_id")
            or row.get("attraction_id")
            or row.get("route_id")
        )
        if row_id == target_id and corrupted_field in row:
            row[corrupted_field] = corrupted_value
            row["attack_annotation"] = attack_mode
            break
    return attacked


def select_attack_for_task(task: TaskSpec, catalog: list[dict], override_mode: str | None = None) -> dict | None:
    for attack in catalog:
        if task.attack_id and attack["attack_id"] == task.attack_id:
            if override_mode and attack["attack_mode"] != override_mode:
                continue
            return attack
    if override_mode:
        for attack in catalog:
            if attack["base_task_id"] == task.task_id and attack["attack_mode"] == override_mode:
                return attack
    return None
