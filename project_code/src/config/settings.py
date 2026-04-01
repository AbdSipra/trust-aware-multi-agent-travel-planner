from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_local_env(root_dir: Path) -> None:
    env_path = root_dir / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _strip_wrapping_quotes(value.strip()))


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    provider: str
    ollama_base_url: str
    ollama_model: str
    groq_api_key: str
    groq_base_url: str
    groq_model: str
    runs_dir: Path
    default_seed: int

    @property
    def project_code_dir(self) -> Path:
        return self.root_dir / "project_code"

    @property
    def data_dir(self) -> Path:
        return self.project_code_dir / "data"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"

    @property
    def tasks_dir(self) -> Path:
        return self.data_dir / "tasks"

    @property
    def attacks_dir(self) -> Path:
        return self.data_dir / "attacks"


def load_settings() -> Settings:
    root_dir = _root_dir()
    _load_local_env(root_dir)
    return Settings(
        root_dir=root_dir,
        provider=os.getenv("AGENTIC_MODEL_PROVIDER", "ollama").strip().lower(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        runs_dir=(root_dir / os.getenv("AGENTIC_RUNS_DIR", "project_code/data/runs")).resolve(),
        default_seed=int(os.getenv("AGENTIC_DEFAULT_SEED", "7")),
    )
