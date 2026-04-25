"""In-memory per-session chat history (last N messages)."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class SessionStore:
    """
    Thread-safe store of recent turns per session.

    Keeps the last ``max_messages`` (user + assistant) entries per ``session_id``.
    For production, replace with Redis or a persistent database.
    """

    max_messages: int
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _sessions: Dict[str, Deque[ChatMessage]] = field(default_factory=dict, repr=False)

    def get_history(self, session_id: str) -> List[ChatMessage]:
        with self._lock:
            d = self._sessions.get(session_id)
            return list(d) if d else []

    def append(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque(maxlen=self.max_messages)
            self._sessions[session_id].append(ChatMessage(role=role, content=content))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def iter_session_ids(self) -> Iterator[str]:
        with self._lock:
            yield from self._sessions


__all__ = ["ChatMessage", "SessionStore"]
