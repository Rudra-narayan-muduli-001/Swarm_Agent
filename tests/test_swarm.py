"""
Swarm Agent — comprehensive test suite.

Covers:
- app/core/agent.py      (schema, handoff, to_text)
- app/sessions.py        (Session, get_session, touch, prune)
- app/core/llm.py        (llm_stream, resolve_model)
- app/core/runtime.py    (run_swarm, _execute, truncation, handoffs)
- app/tools/web_search.py
- app/tools/read_url.py
- app/main.py            (FastAPI routes, SSE streaming)
- app/agents registry
- static assets existence
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_litellm_chunk(content=None, tool_calls=None):
    """Build a fake LiteLLM chunk: chunk.choices[0].delta.content/tool_calls."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk

def make_toolcall_delta(index=0, id=None, name=None, arguments=None):
    tc = MagicMock()
    tc.index = index
    tc.id = id
    func = MagicMock()
    func.name = name
    func.arguments = arguments
    tc.function = func
    return tc

# ---------------------------------------------------------------------------
# 1) app/core/agent.py
# ---------------------------------------------------------------------------

class TestAgentHelpers:
    def test_handoff_with_context(self):
        from app.core.agent import handoff
        assert handoff("researcher", "do X") == {"handoff": "researcher", "context": "do X"}

    def test_handoff_without_context(self):
        from app.core.agent import handoff
        assert handoff("writer") == {"handoff": "writer"}
        assert "context" not in handoff("triage", None)

    def test_is_handoff_true(self):
        from app.core.agent import is_handoff
        assert is_handoff({"handoff": "x"})
        assert is_handoff({"handoff": "x", "context": "y"})

    def test_is_handoff_false(self):
        from app.core.agent import is_handoff
        assert not is_handoff({"error": "oops"})
        assert not is_handoff("handoff")
        assert not is_handoff(None)
        assert not is_handoff({})

    def test_to_text_str(self):
        from app.core.agent import to_text
        assert to_text("hello") == "hello"

    def test_to_text_dict(self):
        from app.core.agent import to_text
        assert json.loads(to_text({"a": 1})) == {"a": 1}

    def test_to_text_non_serializable(self):
        from app.core.agent import to_text
        # set is not JSON serializable -> default=str fallback produces "{}" style via str(set)
        result = to_text({"x", "y"})
        assert isinstance(result, str)

    def test_function_to_schema_simple(self):
        from app.core.agent import function_to_schema
        def greet(name: str, times: int = 1) -> str:
            """Greet someone."""
            return f"hi {name}" * times
        schema = function_to_schema(greet)
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "greet"
        assert fn["description"] == "Greet someone."
        props = fn["parameters"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["times"]["type"] == "integer"
        assert "name" in fn["parameters"]["required"]
        assert "times" not in fn["parameters"]["required"]

    def test_function_to_schema_no_hints_defaults_to_string(self):
        from app.core.agent import function_to_schema
        def foo(a, b):
            """no hints"""
            pass
        schema = function_to_schema(foo)
        assert schema["function"]["parameters"]["properties"]["a"]["type"] == "string"
        assert schema["function"]["parameters"]["properties"]["b"]["type"] == "string"

    def test_function_to_schema_varargs_ignored(self):
        from app.core.agent import function_to_schema
        def foo(a: str, *args, **kwargs):
            """var"""
            pass
        schema = function_to_schema(foo)
        props = schema["function"]["parameters"]["properties"]
        assert "args" not in props
        assert "kwargs" not in props
        assert "a" in props

    def test_function_to_schema_missing_docstring(self):
        from app.core.agent import function_to_schema
        def foo(x: str):
            pass
        schema = function_to_schema(foo)
        assert schema["function"]["description"] == ""

    def test_agent_dataclass_defaults(self):
        from app.core.agent import Agent
        a = Agent(name="test", instructions="hi")
        assert a.tools == []
        assert a.model is None
        assert a.description == ""

    def test_type_map_covers_builtins(self):
        from app.core.agent import _TYPE_MAP
        assert _TYPE_MAP[str] == "string"
        assert _TYPE_MAP[int] == "integer"
        assert _TYPE_MAP[float] == "number"
        assert _TYPE_MAP[bool] == "boolean"


# ---------------------------------------------------------------------------
# 2) app/sessions.py
# ---------------------------------------------------------------------------

class TestSessions:
    def setup_method(self):
        # clear global store before each test
        from app import sessions
        with sessions._lock:
            sessions._sessions.clear()

    def test_session_creation(self):
        from app.sessions import Session
        s = Session()
        assert len(s.id) == 12
        assert s.messages == []
        assert isinstance(s.created_at, float)
        assert isinstance(s.last_active, float)

    def test_get_session_creates_new_when_no_id(self):
        from app.sessions import get_session, _sessions
        s = get_session(None)
        assert s.id in _sessions
        assert len(_sessions) == 1

    def test_get_session_creates_new_when_unknown_id(self):
        from app.sessions import get_session, _sessions
        s1 = get_session("doesnotexist")
        assert s1.id != "doesnotexist"
        assert len(_sessions) == 1

    def test_get_session_reuses_existing(self):
        from app.sessions import get_session
        s1 = get_session(None)
        sid = s1.id
        time.sleep(0.01)
        s2 = get_session(sid)
        assert s1 is s2
        assert s2.last_active >= s1.last_active

    def test_touch_updates_last_active(self):
        from app.sessions import Session, touch
        s = Session()
        old = s.last_active
        time.sleep(0.01)
        touch(s)
        assert s.last_active > old

    def test_prune_removes_stale(self):
        from app.sessions import Session, _sessions, _lock, MAX_AGE_SECONDS, get_session
        # insert a stale session manually
        stale = Session()
        stale.last_active = time.time() - MAX_AGE_SECONDS - 10
        with _lock:
            _sessions[stale.id] = stale
            _sessions["fresh"] = Session()
        # creating a new session triggers prune
        get_session(None)
        with _lock:
            assert stale.id not in _sessions
            assert "fresh" in _sessions

    def test_prune_keeps_fresh(self):
        from app.sessions import get_session, _sessions
        s1 = get_session(None)
        s2 = get_session(None)
        # both should be present (different ids)
        assert len(_sessions) == 2

    def test_session_id_hex_format(self):
        from app.sessions import Session
        s = Session()
        # hex chars only
        assert all(c in "0123456789abcdef" for c in s.id)


# ---------------------------------------------------------------------------
# 3) app/core/llm.py
# ---------------------------------------------------------------------------

class TestLlm:
    def test_resolve_model_uses_agent_override(self):
        from app.core.agent import Agent
        from app.core.llm import resolve_model
        ag = Agent(name="x", instructions="hi", model="custom/model")
        assert resolve_model(ag) == "custom/model"

    def test_resolve_model_falls_back_to_default(self):
        from app.core.agent import Agent
        from app.core.llm import resolve_model, DEFAULT_MODEL
        ag = Agent(name="x", instructions="hi")
        assert resolve_model(ag) == DEFAULT_MODEL

    def test_llm_stream_simple_content(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="triage", instructions="hi")
        chunks = [
            make_litellm_chunk(content="Hello "),
            make_litellm_chunk(content="world"),
            make_litellm_chunk(content=None),
        ]
        on_token_calls = []
        with patch("app.core.llm.completion", return_value=iter(chunks)):
            content, calls = llm_stream(ag, [{"role": "user", "content": "hi"}], [], lambda t: on_token_calls.append(t))
        assert content == "Hello world"
        assert calls == []
        assert on_token_calls == ["Hello ", "world"]

    def test_llm_stream_tool_calls_reassembly(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="researcher", instructions="hi")
        # simulate fragmented tool call: first chunk has id+name+partial args, second has rest
        tc1 = make_toolcall_delta(index=0, id="call_123", name="web_search", arguments='{"query": "hel')
        tc2 = make_toolcall_delta(index=0, id=None, name=None, arguments='lo"}')
        chunks = [
            make_litellm_chunk(tool_calls=[tc1]),
            make_litellm_chunk(tool_calls=[tc2]),
        ]
        with patch("app.core.llm.completion", return_value=iter(chunks)):
            content, calls = llm_stream(ag, [], [{"type":"function","function":{"name":"web_search"}}], lambda t: None)
        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"query": "hello"}
        assert calls[0].id == "call_123"

    def test_llm_stream_multiple_tool_calls(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="researcher", instructions="hi")
        tc1 = make_toolcall_delta(index=0, id="call_1", name="web_search", arguments='{"query":"a"}')
        tc2 = make_toolcall_delta(index=1, id="call_2", name="read_url", arguments='{"url":"http://x"}')
        chunks = [make_litellm_chunk(tool_calls=[tc1]), make_litellm_chunk(tool_calls=[tc2])]
        with patch("app.core.llm.completion", return_value=iter(chunks)):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert len(calls) == 2
        names = {c.name for c in calls}
        assert names == {"web_search", "read_url"}

    def test_llm_stream_generates_id_when_missing(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="triage", instructions="hi")
        tc = make_toolcall_delta(index=0, id="", name="web_search", arguments='{"query":"x"}')
        with patch("app.core.llm.completion", return_value=iter([make_litellm_chunk(tool_calls=[tc])])):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert calls[0].id.startswith("call_")

    def test_llm_stream_invalid_json_args(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="triage", instructions="hi")
        tc = make_toolcall_delta(index=0, id="call_1", name="web_search", arguments='not json')
        with patch("app.core.llm.completion", return_value=iter([make_litellm_chunk(tool_calls=[tc])])):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert calls[0].arguments == {"_raw": "not json"}

    def test_llm_stream_non_dict_json(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="triage", instructions="hi")
        tc = make_toolcall_delta(index=0, id="call_1", name="web_search", arguments='["a","b"]')
        with patch("app.core.llm.completion", return_value=iter([make_litellm_chunk(tool_calls=[tc])])):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert calls[0].arguments == {"_raw": '["a","b"]'}

    def test_llm_stream_sets_groq_api_base(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="triage", instructions="hi", model="groq/llama-3.3-70b-versatile")
        with patch("app.core.llm.completion", return_value=iter([])) as mock_comp, \
             patch("app.core.llm.os.getenv") as mock_getenv:
            # getenv side_effect: return PORT/model etc. Need to handle calls for GROQ_API_BASE and others
            def getenv_side(key, default=None):
                if key == "GROQ_API_BASE":
                    return "https://api.groq.com/openai/v1"
                return default
            mock_getenv.side_effect = getenv_side
            # also need to ensure DEFAULT_MODEL patch? but agent.model overrides
            llm_stream(ag, [], [], lambda t: None)
            # completion called with api_base
            called_kwargs = mock_comp.call_args[1]
            assert called_kwargs["api_base"] == "https://api.groq.com/openai/v1"

    def test_llm_stream_skips_empty_choices(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="triage", instructions="hi")
        bad_chunk = MagicMock()
        bad_chunk.choices = []
        good_chunk = make_litellm_chunk(content="ok")
        with patch("app.core.llm.completion", return_value=iter([bad_chunk, good_chunk])):
            content, _ = llm_stream(ag, [], [], lambda t: None)
        assert content == "ok"

    def test_llm_stream_handles_none_delta(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="triage", instructions="hi")
        chunk = MagicMock()
        choice = MagicMock()
        choice.delta = None
        chunk.choices = [choice]
        with patch("app.core.llm.completion", return_value=iter([chunk, make_litellm_chunk(content="hi")])):
            content, _ = llm_stream(ag, [], [], lambda t: None)
        assert content == "hi"


# ---------------------------------------------------------------------------
# 4) app/core/runtime.py
# ---------------------------------------------------------------------------

class TestRuntime:
    def make_session(self):
        from app.sessions import Session
        s = Session()
        s.messages = []
        return s

    def test_execute_success(self):
        from app.core.runtime import _execute
        def add(a: int, b: int): return a + b
        assert _execute(add, {"a": 2, "b": 3}) == 5

    def test_execute_exception(self):
        from app.core.runtime import _execute
        def boom(x: str): raise ValueError("oops")
        res = _execute(boom, {"x": "hi"})
        assert res == {"error": "ValueError: oops"}

    def test_run_swarm_no_tool_calls(self):
        from app.core.agent import Agent
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        events = []
        def on_event(e, d): events.append((e, d))
        with patch("app.core.runtime.llm_stream", return_value=("final answer", [])):
            run_swarm(session, ag, {"triage": ag}, on_event)
        assert ("agent", {"name": "triage"}) in events
        assert ("done", {}) in events
        assert session.messages[-1] == {"role": "assistant", "content": "final answer"}

    def test_run_swarm_with_tool_call(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def my_tool(query: str) -> str:
            """tool"""
            return f"result for {query}"
        ag = Agent(name="researcher", instructions="sys", tools=[my_tool])
        session = self.make_session()
        # first iteration returns tool call, second returns no calls
        calls = [ToolCall(id="c1", name="my_tool", arguments={"query": "hello"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("thinking", calls), ("done", [])]):
            events = []
            run_swarm(session, ag, {"researcher": ag}, lambda e, d: events.append((e, d)))
        # tool_call and tool_result should be emitted
        assert any(e == "tool_call" and d["tool"] == "my_tool" for e, d in events)
        assert any(e == "tool_result" for e, d in events)
        # messages should contain assistant tool_calls + tool result
        assert any(m.get("tool_calls") for m in session.messages)
        assert any(m.get("role") == "tool" for m in session.messages)

    def test_run_swarm_handoff(self):
        from app.core.agent import Agent, handoff
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def handoff_to_writer(details: str) -> dict:
            """handoff"""
            return handoff("writer", details)
        def handoff_to_triage(details: str) -> dict:
            return handoff("triage")
        triage = Agent(name="triage", instructions="triage", tools=[handoff_to_writer])
        writer = Agent(name="writer", instructions="writer", tools=[handoff_to_triage])
        registry = {"triage": triage, "writer": writer}
        session = self.make_session()
        # triage handoffs to writer, writer returns final
        calls_triage = [ToolCall(id="c1", name="handoff_to_writer", arguments={"details": "go"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls_triage), ("final", [])]):
            events = []
            run_swarm(session, triage, registry, lambda e, d: events.append((e, d)))
        # should have two agent events
        agent_events = [d["name"] for e, d in events if e == "agent"]
        assert agent_events == ["triage", "writer"]
        # tool_result for handoff should say handoff to writer
        assert any(e == "tool_result" and "handoff to writer" in d["result"] for e, d in events)

    def test_run_swarm_unknown_tool(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="unknown_tool", arguments={})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e, d: events.append((e, d)))
        # should still emit tool_result with error string inside messages tool content
        # find tool message
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert any("unknown tool" in m["content"] for m in tool_msgs)

    def test_run_swarm_unknown_handoff_target(self):
        from app.core.agent import Agent, handoff
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def bad_handoff(details: str) -> dict:
            return handoff("nonexistent")
        ag = Agent(name="triage", instructions="sys", tools=[bad_handoff])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="bad_handoff", arguments={"details": "x"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e, d: events.append((e, d)))
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert any("unknown agent" in m["content"] for m in tool_msgs)

    def test_run_swarm_truncation(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm, MAX_TOOL_RESULT_CHARS
        def big_tool(x: str) -> str:
            return "A" * (MAX_TOOL_RESULT_CHARS + 100)
        ag = Agent(name="triage", instructions="sys", tools=[big_tool])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="big_tool", arguments={"x": "y"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e, d: events.append((e, d)))
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert len(tool_msgs[0]["content"]) == MAX_TOOL_RESULT_CHARS + len("\n...[truncated]")
        # tool_result event is truncated to 500
        tr_events = [d for e, d in events if e == "tool_result"]
        assert len(tr_events[0]["result"]) <= 500

    def test_run_swarm_tool_result_event_truncated_to_500(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def t(x: str) -> str: return "B" * 1000
        ag = Agent(name="triage", instructions="sys", tools=[t])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="t", arguments={"x": "y"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e, d: events.append((e, d)))
        tr = [d for e, d in events if e == "tool_result"][0]
        assert len(tr["result"]) == 500

    def test_run_swarm_error_emits_error_and_done(self):
        from app.core.agent import Agent
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        with patch("app.core.runtime.llm_stream", side_effect=RuntimeError("boom")):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e, d: events.append((e, d)))
        assert any(e == "error" and "boom" in d["message"] for e, d in events)
        assert events[-1][0] == "done"

    def test_run_swarm_history_window(self):
        # ensure only last 40 messages are sent to LLM
        from app.core.agent import Agent
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        # fill 50 user messages
        session.messages = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        captured_history = {}
        def fake_llm(agent, history, schemas, on_token):
            captured_history["len"] = len(history)
            # history = system + last 40
            return ("ok", [])
        with patch("app.core.runtime.llm_stream", side_effect=fake_llm):
            run_swarm(session, ag, {"triage": ag}, lambda e, d: None)
        # system + 40 = 41
        assert captured_history["len"] == 41

    def test_run_swarm_last_handoff_wins(self):
        from app.core.agent import Agent, handoff
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def h1(details: str): return handoff("researcher", details)
        def h2(details: str): return handoff("writer", details)
        triage = Agent(name="triage", instructions="sys", tools=[h1, h2])
        researcher = Agent(name="researcher", instructions="sys", tools=[])
        writer = Agent(name="writer", instructions="sys", tools=[])
        registry = {"triage": triage, "researcher": researcher, "writer": writer}
        session = self.make_session()
        calls = [
            ToolCall(id="c1", name="h1", arguments={"details": "a"}),
            ToolCall(id="c2", name="h2", arguments={"details": "b"}),
        ]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("final", [])]):
            events = []
            run_swarm(session, triage, registry, lambda e, d: events.append((e, d)))
        # last handoff wins -> writer
        agent_events = [d["name"] for e, d in events if e == "agent"]
        assert agent_events[-1] == "writer"


# ---------------------------------------------------------------------------
# 5) app/tools/web_search.py
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_web_search_uses_tavily_when_key_set(self):
        from app.tools import web_search as ws_mod
        with patch("app.tools.web_search.os.getenv", return_value="fake_key"), \
             patch("app.tools.web_search._tavily", return_value=[{"title": "t", "url": "u", "snippet": "s"}]) as mock_t, \
             patch("app.tools.web_search._duckduckgo") as mock_ddg:
            result = ws_mod.web_search("hello", max_results=3)
            mock_t.assert_called_once_with("hello", 3)
            mock_ddg.assert_not_called()
            assert "title" in result

    def test_web_search_uses_duckduckgo_when_no_key(self):
        from app.tools import web_search as ws_mod
        with patch("app.tools.web_search.os.getenv", return_value=None), \
             patch("app.tools.web_search._duckduckgo", return_value=[{"title": "t", "url": "u", "snippet": "s"}]) as mock_ddg, \
             patch("app.tools.web_search._tavily") as mock_t:
            result = ws_mod.web_search("hello")
            mock_ddg.assert_called_once()
            mock_t.assert_not_called()
            assert "title" in result

    def test_web_search_clamps_max_results(self):
        from app.tools import web_search as ws_mod
        with patch("app.tools.web_search.os.getenv", return_value=None), \
             patch("app.tools.web_search._duckduckgo", return_value=[{"title":"t","url":"u","snippet":"s"}]) as mock_ddg:
            ws_mod.web_search("q", max_results=100)
            # should be clamped to 10
            assert mock_ddg.call_args[0][1] == 10
            ws_mod.web_search("q", max_results=0)
            assert mock_ddg.call_args[0][1] == 1

    def test_web_search_exception_returns_error_json(self):
        from app.tools import web_search as ws_mod
        with patch("app.tools.web_search.os.getenv", return_value=None), \
             patch("app.tools.web_search._duckduckgo", side_effect=RuntimeError("fail")):
            result = ws_mod.web_search("q")
            data = json.loads(result)
            assert "error" in data
            assert "fail" in data["error"]

    def test_web_search_empty_results_returns_error(self):
        from app.tools import web_search as ws_mod
        with patch("app.tools.web_search.os.getenv", return_value=None), \
             patch("app.tools.web_search._duckduckgo", return_value=[]):
            result = ws_mod.web_search("q")
            data = json.loads(result)
            assert "error" in data

    def test_web_search_truncates_at_max_chars(self):
        from app.tools import web_search as ws_mod
        big = [{"title": "t"*5000, "url": "u", "snippet": "s"*5000}] * 5
        with patch("app.tools.web_search.os.getenv", return_value=None), \
             patch("app.tools.web_search._duckduckgo", return_value=big):
            result = ws_mod.web_search("q")
            assert len(result) <= ws_mod.MAX_CHARS

    def test_duckduckgo_parsing(self):
        from app.tools.web_search import _duckduckgo
        html = """
        <div class="result">
          <a class="result__a" href="https://example.com">Example Title</a>
          <a class="result__snippet">Snippet text here</a>
        </div>
        <div class="result">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fother.com%2Fpage&rut=abc">Other</a>
          <div class="result__snippet">Other snippet</div>
        </div>
        """
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.web_search.httpx.get", return_value=mock_resp):
            results = _duckduckgo("test", 2)
        assert len(results) == 2
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "Snippet text here"
        assert results[1]["url"] == "https://other.com/page"

    def test_tavily_parsing(self):
        from app.tools.web_search import _tavily
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"title": "T", "url": "U", "content": "C"},
                {"title": "T2", "url": "U2", "content": "C2"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.web_search.httpx.post", return_value=mock_resp):
            results = _tavily("q", 2)
        assert results[0] == {"title": "T", "url": "U", "snippet": "C"}

    def test_tavily_sends_correct_payload(self):
        from app.tools.web_search import _tavily
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.web_search.httpx.post", return_value=mock_resp) as mock_post, \
             patch("app.tools.web_search.os.getenv", return_value="key123"):
            _tavily("hello", 5)
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["api_key"] == "key123"
            assert kwargs["json"]["query"] == "hello"
            assert kwargs["json"]["max_results"] == 5


# ---------------------------------------------------------------------------
# 6) app/tools/read_url.py
# ---------------------------------------------------------------------------

class TestReadUrl:
    def test_read_url_success(self):
        from app.tools.read_url import read_url
        html = "<html><body><p>Hello world</p><script>bad</script></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.read_url.httpx.get", return_value=mock_resp):
            text = read_url("https://example.com")
        assert "Hello world" in text
        assert "bad" not in text

    def test_read_url_strips_noise_tags(self):
        from app.tools.read_url import read_url
        html = "<nav>nav</nav><p>keep</p><footer>foot</footer>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.read_url.httpx.get", return_value=mock_resp):
            text = read_url("https://example.com")
        assert "keep" in text
        assert "nav" not in text
        assert "foot" not in text

    def test_read_url_empty_returns_message(self):
        from app.tools.read_url import read_url
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>   </body></html>"
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.read_url.httpx.get", return_value=mock_resp):
            text = read_url("https://example.com")
        assert "No readable text" in text

    def test_read_url_truncates(self):
        from app.tools.read_url import read_url
        long_text = "a" * 10000
        mock_resp = MagicMock()
        mock_resp.text = f"<p>{long_text}</p>"
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.read_url.httpx.get", return_value=mock_resp):
            text = read_url("https://example.com", max_chars=100)
        assert len(text) == 100

    def test_read_url_exception(self):
        from app.tools.read_url import read_url
        with patch("app.tools.read_url.httpx.get", side_effect=RuntimeError("net fail")):
            text = read_url("https://example.com")
        assert "Error reading" in text
        assert "net fail" in text

    def test_read_url_collapses_whitespace(self):
        from app.tools.read_url import read_url
        html = "<p>hello   \n\n  world</p>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch("app.tools.read_url.httpx.get", return_value=mock_resp):
            text = read_url("https://example.com")
        assert text == "hello world"


# ---------------------------------------------------------------------------
# 7) app/main.py  (FastAPI)
# ---------------------------------------------------------------------------

class TestApi:
    def setup_method(self):
        # clear sessions
        from app import sessions
        with sessions._lock:
            sessions._sessions.clear()

    def test_get_index(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "Swarm Agent" in r.text
        # should be the new two-pane UI
        assert "pane-trace" in r.text
        assert "pane-chat" in r.text

    def test_get_agents(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/agents")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data
        names = {a["name"] for a in data["agents"]}
        assert names == {"triage", "researcher", "writer"}

    def test_chat_requires_message(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/chat", params={"message": ""})
        assert r.status_code == 400
        r = client.get("/api/chat", params={"message": "   "})
        assert r.status_code == 400

    def test_chat_unknown_agent(self):
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/chat", params={"message": "hi", "agent": "ghost"})
        assert r.status_code == 404

    def test_chat_streams_simple(self):
        from app.main import app
        client = TestClient(app)

        def fake_run(session, start_agent, registry, on_event):
            on_event("agent", {"name": start_agent.name})
            on_event("token", {"content": "hello "})
            on_event("token", {"content": "world"})
            on_event("done", {})

        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                assert r.status_code == 200
                assert "text/event-stream" in r.headers["content-type"]
                text = "".join(chunk for chunk in r.iter_text())
                assert "event: agent" in text
                assert "event: token" in text
                assert "hello " in text
                assert "event: done" in text

    def test_chat_appends_user_message_to_session(self):
        from app.main import app
        from app.sessions import _sessions
        client = TestClient(app)

        def fake_run(session, start_agent, registry, on_event):
            # verify user message was already appended before run_swarm called
            assert session.messages[-1] == {"role": "user", "content": "my question"}
            on_event("done", {})

        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "my question"}) as r:
                list(r.iter_text())
        # after request, session exists
        assert len(_sessions) == 1
        sess = next(iter(_sessions.values()))
        assert any(m["content"] == "my question" for m in sess.messages)

    def test_chat_session_reuse(self):
        from app.main import app
        from app.sessions import _sessions
        client = TestClient(app)

        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})

        with patch("app.main.run_swarm", side_effect=fake_run):
            # first request creates session
            with client.stream("GET", "/api/chat", params={"message": "first"}) as r:
                list(r.iter_text())
            sid = next(iter(_sessions.values())).id
            msgs_after_first = len(next(iter(_sessions.values())).messages)
            # second request with same sid
            with client.stream("GET", "/api/chat", params={"message": "second", "session_id": sid}) as r:
                list(r.iter_text())
            # same session reused, not new one
            assert len(_sessions) == 1
            assert next(iter(_sessions.values())).id == sid
            assert len(next(iter(_sessions.values())).messages) == msgs_after_first + 1

    def test_chat_sse_headers(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                assert r.headers["Cache-Control"] == "no-cache"
                assert r.headers["X-Accel-Buffering"] == "no"

    def test_sse_stream_handles_tool_events(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("agent", {"name": "researcher"})
            on_event("tool_call", {"tool": "web_search", "arguments": {"query": "x"}, "agent": "researcher"})
            on_event("tool_result", {"tool": "web_search", "result": "ok"})
            on_event("token", {"content": "done"})
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run):
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                text = "".join(r.iter_text())
                assert "event: tool_call" in text
                assert "web_search" in text
                assert "event: tool_result" in text

    def test_touch_called_after_stream(self):
        from app.main import app
        client = TestClient(app)
        def fake_run(session, start_agent, registry, on_event):
            on_event("done", {})
        with patch("app.main.run_swarm", side_effect=fake_run), \
             patch("app.main.touch") as mock_touch:
            with client.stream("GET", "/api/chat", params={"message": "hi"}) as r:
                list(r.iter_text())
            mock_touch.assert_called_once()


# ---------------------------------------------------------------------------
# 8) static assets existence
# ---------------------------------------------------------------------------

class TestStatic:
    def test_static_files_exist(self):
        root = Path(__file__).resolve().parent.parent / "static"
        assert (root / "index.html").exists()
        assert (root / "style.css").exists()
        assert (root / "app.js").exists()

    def test_index_contains_two_pane_markers(self):
        html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
        assert "pane-trace" in html
        assert "pane-chat" in html
        assert "/static/style.css" in html
        assert "/static/app.js" in html

    def test_style_contains_tokens(self):
        css = (Path(__file__).resolve().parent.parent / "static" / "style.css").read_text(encoding="utf-8")
        for tok in ["--honey", "--sage", "--clay", "--bg-0", "--ink-1"]:
            assert tok in css

    def test_app_js_handles_sse(self):
        js = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text(encoding="utf-8")
        for kw in ["EventSource", "tool_call", "tool_result", "agent"]:
            assert kw in js


# ---------------------------------------------------------------------------
# 9) agents registry
# ---------------------------------------------------------------------------

class TestAgentsRegistry:
    def test_registry_contents(self):
        from app.agents import REGISTRY, ALL_AGENTS
        assert len(ALL_AGENTS) == 3
        assert set(REGISTRY.keys()) == {"triage", "researcher", "writer"}

    def test_get_agent(self):
        from app.agents import get_agent
        assert get_agent("triage").name == "triage"
        assert get_agent("ghost") is None

    def test_agents_have_instructions(self):
        from app.agents import ALL_AGENTS
        for ag in ALL_AGENTS:
            assert len(ag.instructions) > 20
            assert ag.description

    def test_handoff_tools_exist(self):
        from app.agents.triage import handoff_to_researcher, handoff_to_writer
        from app.agents.researcher import handoff_to_writer as r_handoff
        assert handoff_to_researcher("x")["handoff"] == "researcher"
        assert handoff_to_writer("x")["handoff"] == "writer"
        assert r_handoff("x")["handoff"] == "writer"
