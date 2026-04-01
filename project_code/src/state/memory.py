from __future__ import annotations

from dataclasses import asdict

from src.state.schemas import MemoryEntry


class SharedMemory:
    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def write(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def accepted_entries(self) -> list[dict]:
        return [asdict(entry) for entry in self._entries if not entry.quarantine_flag]

    def quarantined_entries(self) -> list[dict]:
        return [asdict(entry) for entry in self._entries if entry.quarantine_flag]

    def all_entries(self) -> list[dict]:
        return [asdict(entry) for entry in self._entries]
