from __future__ import annotations

import json
import time
import threading
import warnings

try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except ImportError:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Using `httpx` with `starlette.testclient`.*")

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path


@pytest.fixture(autouse=True)
def clear_sessions():
    from app import sessions
    with sessions._lock:
        sessions._sessions.clear()
    yield
    with sessions._lock:
        sessions._sessions.clear()


class TestSessionsExtended:
    def test_session_slots_restricted(self):
        from app.sessions import Session
        s = Session()
        with pytest.raises(AttributeError):
            s.new_attr = "should fail"

    def test_session_id_unique(self):
        from app.sessions import Session
        ids = {Session().id for _ in range(20)}
        assert len(ids) == 20

    def test_session_id_hex(self):
        from app.sessions import Session
        s = Session()
        assert len(s.id) == 12
        assert all(c in "0123456789abcdef" for c in s.id)

    def test_session_times_are_float(self):
        from app.sessions import Session
        s = Session()
        assert isinstance(s.created_at, float)
        assert isinstance(s.last_active, float)
        assert s.last_active >= s.created_at

    def test_get_session_no_id_creates_new(self):
        from app.sessions import get_session, _sessions
        s = get_session()
        assert s.id in _sessions
        s2 = get_session(None)
        assert s2.id != s.id
        assert len(_sessions) == 2

    def test_get_session_empty_string_creates_new(self):
        from app.sessions import get_session, _sessions
        s = get_session("")
        assert s.id != ""
        assert s.id in _sessions

    def test_get_session_reuses_existing(self):
        from app.sessions import get_session
        s1 = get_session()
        sid = s1.id
        s2 = get_session(sid)
        assert s1 is s2

    def test_get_session_updates_last_active(self):
        from app.sessions import get_session
        s1 = get_session()
        old = s1.last_active
        time.sleep(0.02)
        s2 = get_session(s1.id)
        assert s2.last_active > old

    def test_touch_updates_time(self):
        from app.sessions import Session, touch
        s = Session()
        old = s.last_active
        time.sleep(0.02)
        touch(s)
        assert s.last_active > old

    def test_prune_removes_stale_only(self):
        from app.sessions import Session, _sessions, _lock, MAX_AGE_SECONDS, get_session
        stale = Session()
        stale.last_active = time.time() - MAX_AGE_SECONDS - 100
        fresh = Session()
        with _lock:
            _sessions[stale.id] = stale
            _sessions[fresh.id] = fresh
        get_session()
        with _lock:
            assert stale.id not in _sessions
            assert fresh.id in _sessions

    def test_prune_keeps_all_fresh(self):
        from app.sessions import _sessions, get_session
        for _ in range(5):
            get_session()
        count_before = len(_sessions)
        get_session()
        assert len(_sessions) == count_before + 1

    def test_prune_boundary_not_removed(self):
        from app.sessions import Session, _sessions, _lock, MAX_AGE_SECONDS, get_session
        borderline = Session()
        borderline.last_active = time.time() - MAX_AGE_SECONDS + 10
        with _lock:
            _sessions[borderline.id] = borderline
        get_session()
        with _lock:
            assert borderline.id in _sessions

    def test_concurrent_session_creation(self):
        from app.sessions import get_session, _sessions
        created_ids = []
        errors = []
        def create_session():
            try:
                s = get_session()
                created_ids.append(s.id)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=create_session) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert len(created_ids) == 20
        assert len(set(created_ids)) == 20
        assert len(_sessions) == 20

    def test_concurrent_get_same_session(self):
        from app.sessions import get_session, Session, _sessions, _lock
        base = Session()
        with _lock:
            _sessions[base.id] = base
        sid = base.id
        results = []
        def get_it():
            results.append(get_session(sid))
        threads = [threading.Thread(target=get_it) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert all(r is base for r in results)

    def test_max_age_constant(self):
        from app.sessions import MAX_AGE_SECONDS
        assert MAX_AGE_SECONDS == 3600


class TestAgentsRegistry:
    def test_registry_contents(self):
        from app.agents import REGISTRY, ALL_AGENTS
        assert len(REGISTRY) == 3
        assert len(ALL_AGENTS) == 3
        assert set(REGISTRY.keys()) == {"triage", "researcher", "writer"}

    def test_get_agent_exists(self):
        from app.agents import get_agent
        assert get_agent("triage") is not None
        assert get_agent("researcher") is not None
        assert get_agent("writer") is not None

    def test_get_agent_missing(self):
        from app.agents import get_agent
        assert get_agent("ghost") is None
        assert get_agent("") is None

    def test_agents_instructions(self):
        from app.agents import ALL_AGENTS
        for ag in ALL_AGENTS:
            assert isinstance(ag.instructions, str)
            assert len(ag.instructions) > 50
            assert isinstance(ag.description, str)
            assert len(ag.description) > 5

    def test_triage_has_handoffs(self):
        from app.agents.triage import triage_agent
        names = {f.__name__ for f in triage_agent.tools}
        assert "handoff_to_researcher" in names
        assert "handoff_to_writer" in names

    def test_researcher_has_tools(self):
        from app.agents.researcher import researcher_agent
        names = {f.__name__ for f in researcher_agent.tools}
        assert "web_search" in names
        assert "read_url" in names
        assert "handoff_to_triage" in names

    def test_writer_has_triage_handoff(self):
        from app.agents.writer import writer_agent
        names = {f.__name__ for f in writer_agent.tools}
        assert "handoff_to_triage" in names

    def test_handoff_functions_return_correct_target(self):
        from app.agents.triage import handoff_to_researcher, handoff_to_writer, handoff_to_triage
        assert handoff_to_researcher("details")["handoff"] == "researcher"
        assert handoff_to_writer("details")["handoff"] == "writer"
        assert handoff_to_triage("details")["handoff"] == "triage"

    def test_all_agent_names_unique(self):
        from app.agents import ALL_AGENTS
        names = [a.name for a in ALL_AGENTS]
        assert len(names) == len(set(names))


class TestIntegrationSwarmFlow:
    def make_session(self):
        from app.sessions import Session
        s = Session()
        s.messages = []
        return s

    def test_full_flow_triage_to_researcher_to_writer(self):
        from app.core.agent import Agent, handoff
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def handoff_researcher(details: str): return handoff("researcher", details)
        def handoff_writer(details: str): return handoff("writer", details)
        def handoff_triage(details: str): return handoff("triage", details)
        triage = Agent(name="triage", instructions="triage", tools=[handoff_researcher])
        researcher = Agent(name="researcher", instructions="researcher", tools=[handoff_writer])
        writer = Agent(name="writer", instructions="writer", tools=[handoff_triage])
        registry = {"triage": triage, "researcher": researcher, "writer": writer}
        session = self.make_session()
        calls1 = [ToolCall(id="c1", name="handoff_researcher", arguments={"details": "research X"})]
        calls2 = [ToolCall(id="c2", name="handoff_writer", arguments={"details": "write report"})]
        calls3 = [ToolCall(id="c3", name="handoff_triage", arguments={"details": "done"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls1), ("", calls2), ("", calls3), ("final report", [])]):
            events = []
            run_swarm(session, triage, registry, lambda e,d: events.append((e,d)))
        agent_seq = [d["name"] for e,d in events if e=="agent"]
        assert agent_seq == ["triage", "researcher", "writer", "triage"]
        assert session.messages[-1]["content"] == "final report"

    def test_integration_via_api_with_mocks(self):
        from app.main import app
        from app.core.llm import ToolCall
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("agent", {"name": "triage"})
            on_event("token", {"content": "hello"})
            on_event("tool_call", {"tool": "handoff_to_writer", "arguments": {"details": "go"}, "agent": "triage"})
            on_event("tool_result", {"tool": "handoff_to_writer", "result": "handoff to writer"})
            on_event("agent", {"name": "writer"})
            on_event("token", {"content": " world"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "write something"}) as r:
                assert r.status_code == 200
                text = "".join(r.iter_text())
                assert "hello" in text
                assert "world" in text
                assert "handoff_to_writer" in text

    def test_unknown_tool_does_not_crash_swarm(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="ghost_tool", arguments={})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("recovered", [])]):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e,d: events.append((e,d)))
        assert session.messages[-1]["content"] == "recovered"

    def test_multiple_handoffs_last_wins(self):
        from app.core.agent import Agent, handoff
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def h1(d: str): return handoff("researcher", d)
        def h2(d: str): return handoff("writer", d)
        triage = Agent(name="triage", instructions="sys", tools=[h1, h2])
        researcher = Agent(name="researcher", instructions="sys", tools=[])
        writer = Agent(name="writer", instructions="sys", tools=[])
        registry = {"triage": triage, "researcher": researcher, "writer": writer}
        session = self.make_session()
        calls = [
            ToolCall(id="c1", name="h1", arguments={"d": "a"}),
            ToolCall(id="c2", name="h2", arguments={"d": "b"}),
        ]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("final", [])]):
            events = []
            run_swarm(session, triage, registry, lambda e,d: events.append((e,d)))
        assert [d["name"] for e,d in events if e=="agent"][-1] == "writer"

    def test_sse_stream_survives_no_tokens(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("agent", {"name": "triage"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                assert r.status_code == 200
                text = "".join(r.iter_text())
                assert "event: agent" in text
                assert "event: done" in text


class TestConfigAndEnv:
    def test_default_model_env(self):
        import os
        from importlib import reload
        old = os.environ.get("LLM_MODEL")
        try:
            os.environ["LLM_MODEL"] = "test-model-xyz"
            import app.core.llm as llm_mod
            reload(llm_mod)
            assert llm_mod.DEFAULT_MODEL == "test-model-xyz"
        finally:
            if old is None:
                os.environ.pop("LLM_MODEL", None)
            else:
                os.environ["LLM_MODEL"] = old
            reload(llm_mod)

    def test_temperature_env(self):
        import os
        from importlib import reload
        old = os.environ.get("LLM_TEMPERATURE")
        try:
            os.environ["LLM_TEMPERATURE"] = "0.42"
            import app.core.llm as llm_mod
            reload(llm_mod)
            assert abs(llm_mod.TEMPERATURE - 0.42) < 1e-6
        finally:
            if old is None:
                os.environ.pop("LLM_TEMPERATURE", None)
            else:
                os.environ["LLM_TEMPERATURE"] = old
            reload(llm_mod)

    def test_static_files_exist(self):
        root = Path(__file__).resolve().parent.parent / "static"
        assert (root / "index.html").exists()
        assert (root / "style.css").exists()
        assert (root / "app.js").exists()

    def test_requirements_has_deps(self):
        req = Path(__file__).resolve().parent.parent / "requirements.txt"
        content = req.read_text()
        for dep in ["fastapi", "uvicorn", "httpx", "beautifulsoup4"]:
            assert dep in content.lower()
