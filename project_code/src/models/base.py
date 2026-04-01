from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResponse:
    provider: str
    model: str
    content: str
    raw: dict[str, Any]


class BaseChatModel(ABC):
    provider: str
    model: str

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> ModelResponse:
        raise NotImplementedError
