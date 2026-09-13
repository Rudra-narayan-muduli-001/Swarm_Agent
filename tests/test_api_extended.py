from __future__ import annotations

import json
import time
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


class TestIndexAndStatic:
    def test_index_returns_html(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Swarm Agent" in r.text

    def test_index_contains_app_js_and_css(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/")
        assert "/static/style.css" in r.text
        assert "/static/app.js" in r.text

    def test_static_css_served(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/static/style.css")
        assert r.status_code == 200
        assert "text/css" in r.headers["content-type"] or r.status_code == 200

    def test_static_js_served(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/static/app.js")
        assert r.status_code == 200
        assert "EventSource" in r.text

    def test_static_missing_returns_404(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/static/nonexistent.xyz")
        assert r.status_code == 404

    def test_static_dir_exists(self):
        from app.main import STATIC_DIR
        assert STATIC_DIR.exists()
        assert (STATIC_DIR / "index.html").exists()


class TestAgentsEndpoint:
    def test_list_agents_schema(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/agents")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data
        for ag in data["agents"]:
            assert "name" in ag
            assert "description" in ag
            assert isinstance(ag["name"], str)
            assert len(ag["description"]) > 5

    def test_list_agents_count(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/agents")
        assert len(r.json()["agents"]) == 3

    def test_list_agents_names_exact(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/agents")
        names = {a["name"] for a in r.json()["agents"]}
        assert names == {"triage", "researcher", "writer"}


class TestChatEndpointValidation:
    def test_chat_missing_message_param(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/chat")
        assert r.status_code == 400

    def test_chat_empty_string(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/chat", params={"message": ""})
        assert r.status_code == 400
        assert r.json()["detail"] == "message is required"

    def test_chat_whitespace_only(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/chat", params={"message": "   \t\n  "})
        assert r.status_code == 400

    def test_chat_unknown_agent_returns_404(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/chat", params={"message": "hello", "agent": "nonexistent"})
        assert r.status_code == 404
        assert "unknown agent" in r.json()["detail"]

    def test_chat_valid_agent_param(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            assert start_agent.name == "researcher"
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi", "agent": "researcher"}) as r:
                assert r.status_code == 200
                list(r.iter_text())

    def test_chat_no_agent_uses_first_registry(self):
        from app.main import app
        from app.agents import REGISTRY
        client = TestClient(app)
        first = next(iter(REGISTRY.values())).name
        def fake_run(session, start_agent, registry, on_event):
            assert start_agent.name == first
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                assert r.status_code == 200
                list(r.iter_text())

    def test_chat_message_stripped_before_append(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            assert session.messages[-1]["content"] == "hello world"
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "  hello world  "}) as r:
                list(r.iter_text())


class TestChatSSEHeaders:
    def test_sse_content_type(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                assert r.status_code == 200
                assert "text/event-stream" in r.headers["content-type"]

    def test_sse_headers_no_cache(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                assert r.headers["Cache-Control"] == "no-cache"
                assert r.headers["Connection"] == "keep-alive"
                assert r.headers["X-Accel-Buffering"] == "no"

    def test_sse_event_format_parseable(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("agent", {"name": "triage"})
            on_event("token", {"content": "hi"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                raw = "".join(r.iter_text())
                assert "event: agent\ndata:" in raw
                assert "event: token\ndata:" in raw
                assert "event: done\ndata:" in raw
                for chunk in raw.split("\n\n"):
                    if chunk.strip():
                        lines = chunk.strip().split("\n")
                        assert lines[0].startswith("event: ")
                        assert lines[1].startswith("data: ")
                        json.loads(lines[1][6:])

    def test_sse_tool_events_streamed(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("tool_call", {"tool": "web_search", "arguments": {"query": "test"}, "agent": "researcher"})
            on_event("tool_result", {"tool": "web_search", "result": "ok"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                text = "".join(r.iter_text())
                assert "event: tool_call" in text
                assert "event: tool_result" in text
                assert "web_search" in text

    def test_sse_error_event(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("error", {"message": "boom"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                text = "".join(r.iter_text())
                assert "event: error" in text


class TestChatSessions:
    def test_session_created_on_first_chat(self):
        from app.main import app
        from app.sessions import _sessions
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hello"}) as r:
                list(r.iter_text())
        assert len(_sessions) == 1

    def test_session_reused_with_id(self):
        from app.main import app
        from app.sessions import _sessions
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "first"}) as r:
                list(r.iter_text())
            sid = next(iter(_sessions.values())).id
            with client.stream("GET", "/api/chat", params={"message": "second", "session_id": sid}) as r:
                list(r.iter_text())
            assert len(_sessions) == 1
            assert next(iter(_sessions.values())).id == sid
            assert len(next(iter(_sessions.values())).messages) == 2

    def test_invalid_session_id_creates_new(self):
        from app.main import app
        from app.sessions import _sessions
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi", "session_id": "bogus"}) as r:
                list(r.iter_text())
            assert len(_sessions) == 1
            assert "bogus" not in _sessions

    def test_touch_called_after_stream(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run), patch("app.main.touch") as mock_touch:
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                list(r.iter_text())
            mock_touch.assert_called_once()

    def test_multiple_sequential_runs_same_session(self):
        from app.main import app
        from app.sessions import _sessions
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("token", {"content": "ok"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "msg1"}) as r:
                list(r.iter_text())
            sid = next(iter(_sessions.values())).id
            for i in range(3):
                with client.stream("GET", "/api/chat", params={"message": f"msg{i}", "session_id": sid}) as r:
                    list(r.iter_text())
            assert len(next(iter(_sessions.values())).messages) == 4

    def test_sse_unicode_message(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("token", {"content": "héllo 🌍"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "héllo 🌍 test"}) as r:
                text = "".join(r.iter_text())
                assert "héllo" in text
