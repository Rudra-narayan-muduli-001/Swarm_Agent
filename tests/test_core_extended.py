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
from unittest.mock import patch, MagicMock
from dataclasses import is_dataclass


def make_litellm_chunk(content=None, tool_calls=None):
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


class TestAgentExtended:
    def test_type_map_string_variants(self):
        from app.core.agent import _TYPE_MAP
        assert _TYPE_MAP["str"] == "string"
        assert _TYPE_MAP["int"] == "integer"
        assert _TYPE_MAP["float"] == "number"
        assert _TYPE_MAP["bool"] == "boolean"

    def test_handoff_empty_string_context_not_included(self):
        from app.core.agent import handoff
        assert handoff("writer", "") == {"handoff": "writer"}
        assert "context" not in handoff("writer", "")

    def test_handoff_with_none_context(self):
        from app.core.agent import handoff
        assert handoff("triage", None) == {"handoff": "triage"}

    def test_is_handoff_with_extra_keys(self):
        from app.core.agent import is_handoff
        assert is_handoff({"handoff": "x", "context": "y", "extra": 1})
        assert not is_handoff({"handoff_wrong": "x"})

    def test_to_text_none(self):
        from app.core.agent import to_text
        assert to_text(None) == "null"

    def test_to_text_int(self):
        from app.core.agent import to_text
        assert to_text(42) == "42"

    def test_to_text_list(self):
        from app.core.agent import to_text
        result = to_text([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_to_text_unicode(self):
        from app.core.agent import to_text
        assert to_text("héllo") == "héllo"

    def test_to_text_ensure_ascii_false(self):
        from app.core.agent import to_text
        result = to_text({"key": "héllo 🌍"})
        assert "héllo" in result

    def test_function_to_schema_float_bool(self):
        from app.core.agent import function_to_schema
        def foo(a: float, b: bool, c: str = "hi"):
            pass
        schema = function_to_schema(foo)
        props = schema["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["c"]["type"] == "string"
        assert "a" in schema["function"]["parameters"]["required"]
        assert "b" in schema["function"]["parameters"]["required"]
        assert "c" not in schema["function"]["parameters"]["required"]

    def test_function_to_schema_callable_object(self):
        from app.core.agent import function_to_schema
        class Callable:
            def __call__(self, x: str, y: int = 5):
                pass
        obj = Callable()
        schema = function_to_schema(obj)
        assert schema["function"]["name"] == "Callable"
        assert "x" in schema["function"]["parameters"]["properties"]

    def test_function_to_schema_no_type_annotation_defaults_string(self):
        from app.core.agent import function_to_schema
        def foo(x, y: int):
            pass
        schema = function_to_schema(foo)
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "string"
        assert schema["function"]["parameters"]["properties"]["y"]["type"] == "integer"

    def test_function_to_schema_description_empty_when_no_doc(self):
        from app.core.agent import function_to_schema
        def foo(x: str):
            pass
        schema = function_to_schema(foo)
        assert schema["function"]["description"] == ""

    def test_function_to_schema_unknown_annotation_defaults_string(self):
        from app.core.agent import function_to_schema
        def foo(x: list):
            pass
        schema = function_to_schema(foo)
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "string"

    def test_agent_dataclass_defaults(self):
        from app.core.agent import Agent
        a = Agent(name="x", instructions="hi")
        assert a.tools == []
        assert a.model is None
        assert a.description == ""

    def test_agent_with_tools(self):
        from app.core.agent import Agent
        def tool(x: str): return x
        a = Agent(name="test", instructions="hi", tools=[tool], model="gpt-4", description="desc")
        assert len(a.tools) == 1
        assert a.model == "gpt-4"
        assert a.description == "desc"


class TestLlmExtended:
    def test_resolve_model_override(self):
        from app.core.agent import Agent
        from app.core.llm import resolve_model
        ag = Agent(name="x", instructions="hi", model="groq/llama")
        assert resolve_model(ag) == "groq/llama"

    def test_resolve_model_fallback(self):
        from app.core.agent import Agent
        from app.core.llm import resolve_model, DEFAULT_MODEL
        ag = Agent(name="x", instructions="hi", model=None)
        assert resolve_model(ag) == DEFAULT_MODEL

    def test_resolve_model_empty_string_falls_back(self):
        from app.core.agent import Agent
        from app.core.llm import resolve_model, DEFAULT_MODEL
        ag = Agent(name="x", instructions="hi", model="")
        assert resolve_model(ag) == DEFAULT_MODEL

    def test_llm_stream_includes_tools_when_provided(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        tools = [{"type":"function","function":{"name":"web_search","description":""}}]
        with patch("app.core.llm.completion", return_value=iter([])) as mock:
            llm_stream(ag, [], tools, lambda t: None)
            kwargs = mock.call_args[1]
            assert "tools" in kwargs
            assert kwargs["tool_choice"] == "auto"

    def test_llm_stream_no_tools_no_tool_choice(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        with patch("app.core.llm.completion", return_value=iter([])) as mock:
            llm_stream(ag, [], [], lambda t: None)
            kwargs = mock.call_args[1]
            assert "tools" not in kwargs
            assert "tool_choice" not in kwargs

    def test_llm_stream_non_groq_no_api_base(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi", model="gpt-4o-mini")
        with patch("app.core.llm.completion", return_value=iter([])) as mock, \
             patch("app.core.llm.os.getenv", return_value="https://api.groq.com"):
            llm_stream(ag, [], [], lambda t: None)
            assert "api_base" not in mock.call_args[1]

    def test_llm_stream_groq_no_env_no_api_base(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi", model="groq/llama")
        with patch("app.core.llm.completion", return_value=iter([])) as mock, \
             patch("app.core.llm.os.getenv", return_value=None):
            llm_stream(ag, [], [], lambda t: None)
            assert "api_base" not in mock.call_args[1]

    def test_llm_stream_groq_with_api_base(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi", model="groq/llama-3.3")
        def getenv_side(k, default=None):
            return "https://groq.example" if k == "GROQ_API_BASE" else default
        with patch("app.core.llm.completion", return_value=iter([])) as mock, \
             patch("app.core.llm.os.getenv", side_effect=getenv_side):
            llm_stream(ag, [], [], lambda t: None)
            assert mock.call_args[1]["api_base"] == "https://groq.example"

    def test_llm_stream_content_and_tool_calls_same_chunk(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        tc = make_toolcall_delta(0, "c1", "web_search", '{"query":"hi"}')
        chunk = make_litellm_chunk(content="thinking ", tool_calls=[tc])
        on_tokens = []
        with patch("app.core.llm.completion", return_value=iter([chunk])):
            content, calls = llm_stream(ag, [], [], lambda t: on_tokens.append(t))
        assert content == "thinking "
        assert len(calls) == 1
        assert on_tokens == ["thinking "]

    def test_llm_stream_fragmented_three_chunks(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        tc1 = make_toolcall_delta(0, "c1", "web_", arguments=None)
        tc2 = make_toolcall_delta(0, None, "search", arguments='{"q":"')
        tc3 = make_toolcall_delta(0, None, None, arguments='hello"}')
        chunks = [make_litellm_chunk(tool_calls=[tc1]), make_litellm_chunk(tool_calls=[tc2]), make_litellm_chunk(tool_calls=[tc3])]
        with patch("app.core.llm.completion", return_value=iter(chunks)):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"q": "hello"}

    def test_llm_stream_empty_arguments_becomes_empty_dict(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        tc = make_toolcall_delta(0, "c1", "web_search", "")
        with patch("app.core.llm.completion", return_value=iter([make_litellm_chunk(tool_calls=[tc])])):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert calls[0].arguments == {}

    def test_llm_stream_index_none_defaults_to_zero(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        tc = make_toolcall_delta(None, "c1", "web_search", '{"x":1}')
        with patch("app.core.llm.completion", return_value=iter([make_litellm_chunk(tool_calls=[tc])])):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert len(calls) == 1

    def test_llm_stream_skips_none_delta_and_empty_choices(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        bad1 = MagicMock()
        bad1.choices = []
        bad2 = MagicMock()
        bad2.choices = [MagicMock(delta=None)]
        good = make_litellm_chunk(content="ok")
        with patch("app.core.llm.completion", return_value=iter([bad1, bad2, good])):
            content, calls = llm_stream(ag, [], [], lambda t: None)
        assert content == "ok"
        assert calls == []

    def test_llm_stream_toolcall_with_no_function(self):
        from app.core.llm import llm_stream
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        tc = MagicMock()
        tc.index = 0
        tc.id = "c1"
        tc.function = None
        chunk = make_litellm_chunk(tool_calls=[tc])
        with patch("app.core.llm.completion", return_value=iter([chunk])):
            _, calls = llm_stream(ag, [], [], lambda t: None)
        assert len(calls) == 1
        assert calls[0].name == ""
        assert calls[0].arguments == {}

    def test_llm_stream_temperature_and_model_passed(self):
        from app.core.llm import llm_stream, TEMPERATURE, DEFAULT_MODEL
        from app.core.agent import Agent
        ag = Agent(name="x", instructions="hi")
        with patch("app.core.llm.completion", return_value=iter([])) as mock:
            llm_stream(ag, [{"role":"user","content":"hi"}], [], lambda t: None)
            kwargs = mock.call_args[1]
            assert kwargs["model"] == DEFAULT_MODEL
            assert kwargs["temperature"] == TEMPERATURE
            assert kwargs["stream"] is True
            assert kwargs["messages"] == [{"role":"user","content":"hi"}]


class TestRuntimeExtended:
    def make_session(self):
        from app.sessions import Session
        s = Session()
        s.messages = []
        return s

    def test_tools_by_name(self):
        from app.core.runtime import _tools_by_name
        from app.core.agent import Agent
        def a(x: str): pass
        def b(y: int): pass
        ag = Agent(name="x", instructions="hi", tools=[a,b])
        mapping = _tools_by_name(ag)
        assert mapping["a"] is a
        assert mapping["b"] is b

    def test_execute_with_kwargs_error(self):
        from app.core.runtime import _execute
        def foo(a: str, b: str): return a+b
        res = _execute(foo, {"a": "hi"})
        assert "error" in res
        assert "TypeError" in res["error"]

    def test_execute_raises_custom_exception(self):
        from app.core.runtime import _execute
        def boom(): raise RuntimeError("custom")
        res = _execute(boom, {})
        assert res == {"error": "RuntimeError: custom"}

    def test_run_swarm_empty_content_no_append(self):
        from app.core.agent import Agent
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        events = []
        with patch("app.core.runtime.llm_stream", return_value=("", [])):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: events.append((e,d)))
        assert len(session.messages) == 0
        assert events[-1][0] == "done"

    def test_run_swarm_none_content_no_append(self):
        from app.core.agent import Agent
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        events = []
        with patch("app.core.runtime.llm_stream", return_value=(None, [])):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: events.append((e,d)))
        assert session.messages == []

    def test_run_swarm_mixed_handoff_and_normal_tool(self):
        from app.core.agent import Agent, handoff
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def handoff_fn(details: str): return handoff("writer")
        def normal_tool(x: str): return f"normal:{x}"
        triage = Agent(name="triage", instructions="sys", tools=[handoff_fn, normal_tool])
        writer = Agent(name="writer", instructions="sys", tools=[])
        registry = {"triage": triage, "writer": writer}
        session = self.make_session()
        calls = [
            ToolCall(id="c1", name="normal_tool", arguments={"x": "hello"}),
            ToolCall(id="c2", name="handoff_fn", arguments={"details": "go"}),
        ]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("final", [])]):
            events = []
            run_swarm(session, triage, registry, lambda e,d: events.append((e,d)))
        assert any(e=="tool_call" for e,d in events)
        assert any(e=="tool_result" for e,d in events)
        agent_names = [d["name"] for e,d in events if e=="agent"]
        assert agent_names[-1] == "writer"

    def test_run_swarm_truncation_boundary(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm, MAX_TOOL_RESULT_CHARS
        def exact_tool(x: str): return "A" * MAX_TOOL_RESULT_CHARS
        ag = Agent(name="triage", instructions="sys", tools=[exact_tool])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="exact_tool", arguments={"x":"y"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: None)
        tool_msg = [m for m in session.messages if m.get("role")=="tool"][0]
        assert len(tool_msg["content"]) == MAX_TOOL_RESULT_CHARS
        assert "...[truncated]" not in tool_msg["content"]

    def test_run_swarm_truncation_plus_one(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm, MAX_TOOL_RESULT_CHARS
        def big(x: str): return "A" * (MAX_TOOL_RESULT_CHARS + 1)
        ag = Agent(name="triage", instructions="sys", tools=[big])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="big", arguments={"x":"y"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: None)
        tool_msg = [m for m in session.messages if m.get("role")=="tool"][0]
        assert tool_msg["content"].endswith("\n...[truncated]")
        assert len(tool_msg["content"]) == MAX_TOOL_RESULT_CHARS + len("\n...[truncated]")

    def test_run_swarm_tool_result_event_capped_500(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def t(x: str): return "B"*2000
        ag = Agent(name="triage", instructions="sys", tools=[t])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="t", arguments={"x":"y"})]
        events = []
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: events.append((e,d)))
        tr = [d for e,d in events if e=="tool_result"][0]
        assert len(tr["result"]) == 500

    def test_run_swarm_history_window_includes_system(self):
        from app.core.agent import Agent
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="system prompt", tools=[])
        session = self.make_session()
        session.messages = [{"role":"user","content": f"msg {i}"} for i in range(5)]
        captured = {}
        def fake_llm(agent, history, schemas, on_token):
            captured["history"] = history
            return ("ok", [])
        with patch("app.core.runtime.llm_stream", side_effect=fake_llm):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: None)
        assert captured["history"][0] == {"role": "system", "content": "system prompt"}
        assert len(captured["history"]) == 6

    def test_run_swarm_multiple_iterations(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def tool1(x: str): return "r1"
        def tool2(x: str): return "r2"
        ag = Agent(name="triage", instructions="sys", tools=[tool1, tool2])
        session = self.make_session()
        calls1 = [ToolCall(id="c1", name="tool1", arguments={"x":"a"})]
        calls2 = [ToolCall(id="c2", name="tool2", arguments={"x":"b"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls1), ("", calls2), ("final", [])]):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e,d: events.append((e,d)))
        assert events.count(("done", {})) == 1 or any(e=="done" for e,d in events)
        assert session.messages[-1] == {"role": "assistant", "content": "final"}

    def test_run_swarm_error_still_sends_done(self):
        from app.core.agent import Agent
        from app.core.runtime import run_swarm
        ag = Agent(name="triage", instructions="sys", tools=[])
        session = self.make_session()
        events = []
        with patch("app.core.runtime.llm_stream", side_effect=ValueError("oops")):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: events.append((e,d)))
        assert any(e=="error" for e,d in events)
        assert events[-1][0] == "done"

    def test_run_swarm_handoff_unknown_target_error(self):
        from app.core.agent import Agent, handoff
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def bad(details: str): return handoff("ghost")
        ag = Agent(name="triage", instructions="sys", tools=[bad])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="bad", arguments={"details":"x"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            events = []
            run_swarm(session, ag, {"triage": ag}, lambda e,d: events.append((e,d)))
        tool_msgs = [m for m in session.messages if m.get("role")=="tool"]
        assert "unknown agent" in tool_msgs[0]["content"]

    def test_run_swarm_tool_returns_dict_serialized(self):
        from app.core.agent import Agent
        from app.core.llm import ToolCall
        from app.core.runtime import run_swarm
        def returns_dict(x: str): return {"result": "ok", "value": 123}
        ag = Agent(name="triage", instructions="sys", tools=[returns_dict])
        session = self.make_session()
        calls = [ToolCall(id="c1", name="returns_dict", arguments={"x":"y"})]
        with patch("app.core.runtime.llm_stream", side_effect=[("", calls), ("", [])]):
            run_swarm(session, ag, {"triage": ag}, lambda e,d: None)
        tool_msg = [m for m in session.messages if m.get("role")=="tool"][0]
        data = json.loads(tool_msg["content"])
        assert data["result"] == "ok"
