from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.settings import Settings
from src.utils.io import read_csv, read_json


@dataclass
class KnowledgeStore:
    cities: list[dict]
    flights: list[dict]
    hotels: list[dict]
    attractions: list[dict]
    routes: list[dict]

    @classmethod
    def from_settings(cls, settings: Settings) -> "KnowledgeStore":
        knowledge_dir = settings.knowledge_dir
        return cls(
            cities=read_json(knowledge_dir / "cities.json"),
            flights=read_csv(knowledge_dir / "flights.csv"),
            hotels=read_csv(knowledge_dir / "hotels.csv"),
            attractions=read_csv(knowledge_dir / "attractions.csv"),
            routes=read_csv(knowledge_dir / "routes.csv"),
        )

    @classmethod
    def from_dir(cls, directory: Path) -> "KnowledgeStore":
        return cls(
            cities=read_json(directory / "cities.json"),
            flights=read_csv(directory / "flights.csv"),
            hotels=read_csv(directory / "hotels.csv"),
            attractions=read_csv(directory / "attractions.csv"),
            routes=read_csv(directory / "routes.csv"),
        )

    def city_by_id(self, city_id: str) -> dict:
        for city in self.cities:
            if city["city_id"] == city_id:
                return city
        raise KeyError(f"Unknown city_id: {city_id}")
