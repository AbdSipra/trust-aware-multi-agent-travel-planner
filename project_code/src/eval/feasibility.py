from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.state.schemas import TaskSpec


def stay_nights(task: TaskSpec) -> int:
    start = date.fromisoformat(task.trip_start_date)
    end = date.fromisoformat(task.trip_end_date)
    return max(1, (end - start).days)


def room_count_for_task(task: TaskSpec) -> int:
    return max(1, task.traveler_count // 2)


def compute_total_cost(
    task: TaskSpec,
    selected_flight: dict | None,
    selected_hotel: dict | None,
    selected_attractions: list[dict],
) -> dict[str, float]:
    nights = stay_nights(task)
    room_count = room_count_for_task(task)
    flight_cost = float(selected_flight["price_usd"]) * task.traveler_count if selected_flight else 0.0
    hotel_cost = float(selected_hotel["price_per_night_usd"]) * nights * room_count if selected_hotel else 0.0
    attraction_cost = sum(float(item["ticket_price_usd"]) for item in selected_attractions) * task.traveler_count
    total_cost = round(flight_cost + hotel_cost + attraction_cost, 2)
    return {
        "flight_cost_usd": round(flight_cost, 2),
        "hotel_cost_usd": round(hotel_cost, 2),
        "attraction_cost_usd": round(attraction_cost, 2),
        "total_cost_usd": total_cost,
    }


def _candidate_flights(task: TaskSpec, flights: list[dict]) -> list[dict]:
    matches = [
        row
        for row in flights
        if row["origin_city_id"] == task.origin_city_id
        and row["destination_city_id"] == task.destination_city_id
        and row["departure_date"] == task.trip_start_date
        and int(row["stops"]) <= task.max_stops
        and int(row["seats_available"]) >= task.traveler_count
        and row["arrival_time"] <= task.must_arrive_before
        and row["departure_time"] >= task.must_depart_after
        and bool(row.get("baggage_included"))
    ]
    matches.sort(key=lambda row: (float(row["price_usd"]), int(row["stops"]), row["departure_time"]))
    return matches


def _candidate_hotels(task: TaskSpec, hotels: list[dict]) -> list[dict]:
    room_count = room_count_for_task(task)
    matches = [
        row
        for row in hotels
        if row["city_id"] == task.destination_city_id
        and float(row["star_rating"]) >= task.hotel_min_rating
        and row["check_in_date"] <= task.trip_start_date
        and row["check_out_date"] >= task.trip_end_date
        and int(row["rooms_available"]) >= room_count
        and (not task.hard_constraints.get("require_breakfast") or row.get("breakfast_included") == "yes")
    ]
    matches.sort(
        key=lambda row: (
            float(row["price_per_night_usd"]),
            float(row["distance_from_center_km"]),
            -float(row["star_rating"]),
        )
    )
    return matches


def _choose_min_cost_attractions(task: TaskSpec, attractions: list[dict]) -> list[dict]:
    destination_attractions = [
        row for row in attractions if row["city_id"] == task.destination_city_id and row["category"] in set(task.must_visit_categories)
    ]
    by_category: dict[str, list[dict]] = {}
    for category in task.must_visit_categories:
        matches = [row for row in destination_attractions if row["category"] == category]
        matches.sort(key=lambda row: (float(row["ticket_price_usd"]), -float(row["popularity_score"])))
        if matches:
            by_category[category] = matches

    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    for category in task.must_visit_categories:
        options = by_category.get(category, [])
        if not options:
            continue
        selection = options[0]
        if selection["attraction_id"] not in chosen_ids:
            chosen.append(selection)
            chosen_ids.add(selection["attraction_id"])

    if chosen:
        return chosen

    fallback = [row for row in attractions if row["city_id"] == task.destination_city_id]
    fallback.sort(key=lambda row: (float(row["ticket_price_usd"]), -float(row["popularity_score"])))
    return fallback[:1]


def find_min_feasible_package(
    task: TaskSpec,
    flights: list[dict],
    hotels: list[dict],
    attractions: list[dict],
) -> dict[str, Any] | None:
    candidate_flights = _candidate_flights(task, flights)
    candidate_hotels = _candidate_hotels(task, hotels)
    candidate_attractions = _choose_min_cost_attractions(task, attractions)

    if not candidate_flights or not candidate_hotels:
        return None

    best_package: dict[str, Any] | None = None
    for flight in candidate_flights[: min(5, len(candidate_flights))]:
        for hotel in candidate_hotels[: min(8, len(candidate_hotels))]:
            costs = compute_total_cost(task, flight, hotel, candidate_attractions)
            package = {
                "selected_flight": flight,
                "selected_hotel": hotel,
                "selected_attractions": candidate_attractions,
                "budget": costs,
            }
            if best_package is None or costs["total_cost_usd"] < best_package["budget"]["total_cost_usd"]:
                best_package = package
    return best_package


def update_query_budget(user_query: str, budget_limit_usd: float) -> str:
    replacement = f"${int(budget_limit_usd)}" if float(budget_limit_usd).is_integer() else f"${budget_limit_usd:.2f}"
    return re.sub(r"\$\d+(?:\.\d+)?", replacement, user_query, count=1)


def rebalance_tasks_to_feasibility(
    task_records: list[dict],
    flights: list[dict],
    hotels: list[dict],
    attractions: list[dict],
    headroom_ratio: float = 0.08,
    min_headroom_usd: float = 40.0,
) -> tuple[list[dict], list[dict[str, Any]]]:
    adjusted: list[dict] = []
    audit_rows: list[dict[str, Any]] = []

    for record in task_records:
        task = TaskSpec(**record)
        package = find_min_feasible_package(task, flights, hotels, attractions)
        updated = dict(record)
        audit_row: dict[str, Any] = {
            "task_id": task.task_id,
            "old_budget_limit_usd": float(task.budget_limit_usd),
            "status": "no_feasible_package",
        }

        if package is not None:
            min_total = package["budget"]["total_cost_usd"]
            required_budget = round(min_total + max(min_headroom_usd, min_total * headroom_ratio), 2)
            if updated["budget_limit_usd"] < required_budget:
                updated["budget_limit_usd"] = required_budget
                updated["user_query"] = update_query_budget(updated["user_query"], required_budget)
                audit_row["status"] = "budget_raised"
            else:
                audit_row["status"] = "already_feasible"
            audit_row["min_feasible_total_usd"] = min_total
            audit_row["required_budget_usd"] = required_budget
            audit_row["selected_flight_id"] = package["selected_flight"]["flight_id"]
            audit_row["selected_hotel_id"] = package["selected_hotel"]["hotel_id"]
            audit_row["selected_attraction_ids"] = [
                item["attraction_id"] for item in package["selected_attractions"]
            ]
            audit_row["new_budget_limit_usd"] = float(updated["budget_limit_usd"])

        adjusted.append(updated)
        audit_rows.append(audit_row)

    return adjusted, audit_rows
