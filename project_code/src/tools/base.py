from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.state.schemas import ToolObservation


class BaseTool(ABC):
    name: str

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolObservation:
        raise NotImplementedError
