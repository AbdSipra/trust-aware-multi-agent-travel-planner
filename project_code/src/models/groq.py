from __future__ import annotations

import json
from urllib import error, request

from src.models.base import BaseChatModel, ModelResponse


class GroqChatModel(BaseChatModel):
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for GroqChatModel.")
        self.provider = "groq"
        self.api_key = api_key
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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"} if json_mode else {"type": "text"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Groq request failed: {exc}") from exc
        content = raw["choices"][0]["message"]["content"]
        return ModelResponse(provider=self.provider, model=self.model, content=content, raw=raw)
