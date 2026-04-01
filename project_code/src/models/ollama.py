from __future__ import annotations

import json
from urllib import error, request

from src.models.base import BaseChatModel, ModelResponse


class OllamaChatModel(BaseChatModel):
    def __init__(self, base_url: str, model: str) -> None:
        self.provider = "ollama"
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        content = raw.get("message", {}).get("content", "")
        return ModelResponse(provider=self.provider, model=self.model, content=content, raw=raw)
