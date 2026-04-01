from __future__ import annotations

from urllib import error, request

from src.config.settings import Settings
from src.models.base import BaseChatModel
from src.models.groq import GroqChatModel
from src.models.ollama import OllamaChatModel


def _ollama_available(base_url: str) -> bool:
    req = request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with request.urlopen(req, timeout=1.5):
            return True
    except (error.URLError, TimeoutError):
        return False


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    if settings.provider == "groq":
        if not settings.groq_api_key:
            return None
        return GroqChatModel(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            model=settings.groq_model,
        )
    if settings.provider == "ollama":
        if not _ollama_available(settings.ollama_base_url):
            return None
        return OllamaChatModel(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    return None
