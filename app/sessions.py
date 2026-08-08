"""In-memory chat sessions. Idempotent GET-style API, so a session is
created lazily and reused when the client passes its id back."""

from __future__ import annotations

import threading
import time
import uuid

MAX_AGE_SECONDS = 3600


class Session:
    __slots__ = ("id", "messages", "created_at", "last_active")

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.messages: list[dict] = []
        self.created_at = time.time()
        self.last_active = time.time()


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def get_session(session_id: str | None = None) -> Session:
    with _lock:
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session.last_active = time.time()
            return session
        session = Session()
        _sessions[session.id] = session
        _prune_unlocked()
        return session


def touch(session: Session) -> None:
    session.last_active = time.time()


def _prune_unlocked() -> None:
    now = time.time()
    stale = [sid for sid, s in _sessions.items() if now - s.last_active > MAX_AGE_SECONDS]
    for sid in stale:
        del _sessions[sid]
