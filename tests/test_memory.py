"""SQA: in-memory session store — app.memory.store."""

from __future__ import annotations

import threading

import pytest

from app.memory.store import SessionStore


def test_session_empty_history() -> None:
    s = SessionStore(max_messages=8)
    assert s.get_history("unknown") == []


def test_session_round_trip() -> None:
    s = SessionStore(max_messages=8)
    s.append("u1", "user", "Hello")
    s.append("u1", "assistant", "Hi there")
    h = s.get_history("u1")
    assert len(h) == 2
    assert h[0] == {"role": "user", "content": "Hello"}
    assert h[1] == {"role": "assistant", "content": "Hi there"}


def test_session_respects_max_messages() -> None:
    s = SessionStore(max_messages=4)
    for i in range(6):
        s.append("s", "user", f"u{i}")
        s.append("s", "assistant", f"a{i}")
    h = s.get_history("s")
    assert len(h) == 4
    # Oldest evicted: last four messages
    assert h[0]["content"] == "u4"
    assert h[-1]["content"] == "a5"


def test_session_clear() -> None:
    s = SessionStore(max_messages=8)
    s.append("k", "user", "x")
    s.clear("k")
    assert s.get_history("k") == []


def test_session_isolated_by_id() -> None:
    s = SessionStore(max_messages=8)
    s.append("a", "user", "one")
    s.append("b", "user", "two")
    assert s.get_history("a")[0]["content"] == "one"
    assert s.get_history("b")[0]["content"] == "two"


def test_session_concurrent_appends() -> None:
    s = SessionStore(max_messages=200)
    errors: list[BaseException] = []

    def worker(sid: str) -> None:
        try:
            for i in range(100):
                s.append(sid, "user", str(i))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("c1",))
    t2 = threading.Thread(target=worker, args=("c1",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors
    assert len(s.get_history("c1")) == 200


def test_iter_session_ids() -> None:
    s = SessionStore(max_messages=4)
    s.append("d", "user", "x")
    assert list(s.iter_session_ids()) == ["d"]
