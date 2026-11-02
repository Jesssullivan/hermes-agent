"""Route custody between the requests of a turn (agent/route_hooks.py ``check_request_route``).

CADE-1 review 5728175100 (B3): after a failover, a tool row stamped with ``data_use`` reached the
fallback on the next request, and the gate had been asked once with ``data_use`` empty. F12 widens
the check to every request on any route (``test_route_hooks_serving_route.py``). Each probe drives
the real ``run_conversation`` loop with a stub provider call and a stub tool. No provider is called."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import route_hooks
from agent.message_metadata import stamp_data_use
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent

HOSTED = {"provider": "zai", "model": "glm-4.7", "api_key": "fb-secret"}
RAW = "RAW-CLIENT-ROW"
TOOL_DATA = "acme:client-raw"


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture(autouse=True)
def _fresh_route_context(monkeypatch):
    from agent import retry_utils
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **k: 0.0)
    token = route_hooks._ROUTE_SCOPE.set(None)
    with route_hooks._SESSIONS_LOCK:
        saved = dict(route_hooks._SESSIONS)
        route_hooks._SESSIONS.clear()
    yield
    with route_hooks._SESSIONS_LOCK:
        route_hooks._SESSIONS.clear()
        route_hooks._SESSIONS.update(saved)
    route_hooks._ROUTE_SCOPE.reset(token)


def _response(content="done", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_call():
    return SimpleNamespace(id="call_1", type="function", call_id=None, response_item_id=None,
                           function=SimpleNamespace(name="web_search", arguments="{}"))


class _Turn:
    """A turn whose primary answers 429, whose fallback calls one tool, and whose tool stamps data."""

    def __init__(self, chain=(HOSTED,), *, fail_primary=True):
        with (
            patch("model_tools.get_tool_definitions", return_value=[
                {"type": "function", "function": {"name": "web_search", "description": "s",
                                                  "parameters": {"type": "object", "properties": {}}}}]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            self.agent = AIAgent(api_key="primary-key", base_url="https://openrouter.ai/api/v1", provider="openrouter",
                                 model="gpt-4o", quiet_mode=True, skip_context_files=True, skip_memory=True,
                                 fallback_model=list(chain))
        agent = self.agent
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = agent.save_trajectories = False
        agent.compression_enabled = False
        agent._disable_streaming = True
        self.primary, self.fail_primary = agent.provider, fail_primary
        self.sent, self.tool_calls = [], 0

    def send(self, api_kwargs, **_kw):
        agent = self.agent
        raw = RAW in json.dumps(api_kwargs.get("messages") or [], default=str)
        self.sent.append((agent.provider, raw))
        if agent.provider == self.primary and self.fail_primary and not raw:
            raise _rate_limit()
        return _response() if raw else _response("", [_tool_call()])

    def tools(self, assistant_message, messages, *_a, **_kw):
        self.tool_calls += 1
        row = {"role": "tool", "name": "web_search", "tool_call_id": assistant_message.tool_calls[0].id, "content": RAW}
        stamp_data_use(row, [TOOL_DATA])
        messages.append(row)

    def run(self):
        agent = self.agent
        client = (MagicMock(base_url="https://api.z.ai/v1", api_key="fb-secret"), "resolved")
        with patch.object(agent, "_interruptible_api_call", side_effect=self.send), \
                patch.object(agent, "_execute_tool_calls", side_effect=self.tools), \
                patch.object(agent, "_persist_session"), patch.object(agent, "_cleanup_task_resources"), \
                patch("agent.auxiliary_client.resolve_provider_client", return_value=client), \
                patch("agent.chat_completion_helpers._rebuild_fallback_client", create=True):
            return agent.run_conversation("look that up")


def _rate_limit():
    error = Exception("rate limited")
    error.status_code = 429
    return error


def _refuse_tool_data(asked, *, refuse=True, on="zai"):
    """Refuses the tool data on *on*, the hosted route. Only that route's calls are recorded."""
    def custody(phase, candidate, data_use, **_kw):
        if candidate["provider"] != on:
            return {"action": "allow"}
        asked.append((phase, candidate["provider"], list(data_use)))
        if refuse and TOOL_DATA in data_use:
            return {"action": "skip", "reason": "client data"}
        return {"action": "allow"}
    return custody


class TestFailoverThenToolRow:
    def test_a_refused_row_never_reaches_the_fallback(self, hooks):
        asked = []
        hooks["route_custody"] = [_refuse_tool_data(asked)]
        turn = _Turn()
        result = turn.run()
        assert ("zai", False) in turn.sent and ("zai", True) not in turn.sent
        assert turn.sent[-1] == (turn.primary, True), "the restored primary carries the row"
        assert asked[0] == ("failover", "zai", []) and asked[-1] == ("request", "zai", [TOOL_DATA])
        assert {phase for phase, *_rest in asked} == {"failover", "request"}
        assert result["final_response"] == "done"
        assert not turn.agent._provider_fallback_active

    def test_no_primary_to_restore_stops_the_turn_with_the_row_kept(self, hooks):
        asked = []
        hooks["route_custody"] = [_refuse_tool_data(asked)]
        turn = _Turn()
        with patch.object(turn.agent, "_restore_primary_runtime", return_value=False):
            result = turn.run()
        assert ("zai", True) not in turn.sent
        assert result["failed"] is True and result["failure_reason"] == "route_custody"
        assert result["turn_exit_reason"] == "route_custody" and result["failure_retryable"] is False
        assert any(m.get("content") == RAW for m in result["messages"])

    def test_the_gate_reply_replaces_a_stopped_turn(self, hooks):
        hooks["route_custody"] = [_refuse_tool_data([])]
        events = []
        hooks["on_routes_exhausted"] = [lambda **kw: events.append(kw) or {"reply": "held for review"}]
        turn = _Turn()
        with patch.object(turn.agent, "_restore_primary_runtime", return_value=False):
            result = turn.run()
        assert result["final_response"] == "held for review"
        assert [event["reason"] for event in events] == ["route_custody"]

    def test_an_admitted_row_is_asked_each_request_and_the_turn_goes_on(self, hooks):
        asked = []
        hooks["route_custody"] = [_refuse_tool_data(asked, refuse=False)]
        turn = _Turn()
        result = turn.run()
        assert turn.sent[-1] == ("zai", True) and result["final_response"] == "done"
        assert asked[0] == ("failover", "zai", []) and asked[-1] == ("request", "zai", [TOOL_DATA])

    def test_no_gate_and_no_key_never_asks(self, hooks):
        turn = _Turn()
        with patch("agent.route_hooks._invoke", side_effect=AssertionError("no hook may run")):
            result = turn.run()
        assert turn.sent[-1] == ("zai", True) and result["final_response"] == "done"


class TestResolutionEntry:
    def test_a_refused_row_stops_a_turn_built_on_the_entry(self, hooks):
        asked = []
        hooks["route_custody"] = [_refuse_tool_data(asked, on="openrouter")]
        turn = _Turn(chain=(), fail_primary=False)
        route_hooks.bind_resolution_entry(turn.agent, route_hooks.resolution_record(
            {"provider": turn.primary, "model": turn.agent.model}, {"provider": turn.primary}, model=turn.agent.model))
        result = turn.run()
        assert turn.sent == [(turn.primary, False)], "the row never left"
        assert result["failure_reason"] == "route_custody"
        assert [(phase, data_use) for phase, _p, data_use in asked] == [("turn", []), ("request", []),
                                                                        ("request", [TOOL_DATA])]


class TestPostToolCompaction:
    def _compress_spy(self, turn, calls):
        def compress(*_a, messages=None, **kwargs):
            calls.append(turn.agent.provider)
            return SimpleNamespace(
                messages=messages, active_system_prompt=kwargs["active_system_prompt"],
                conversation_history=kwargs["conversation_history"], compression_attempts=kwargs["compression_attempts"],
                final_response=kwargs["final_response"], turn_exit_reason=kwargs["turn_exit_reason"],
                current_turn_user_idx=kwargs["current_turn_user_idx"], end_turn=False)
        return patch("agent.turn_tool_round.compress_after_tool_results", side_effect=compress)

    def test_a_stopped_turn_runs_no_compaction(self, hooks):
        hooks["route_custody"] = [_refuse_tool_data([])]
        turn, calls = _Turn(), []
        with self._compress_spy(turn, calls), patch.object(turn.agent, "_restore_primary_runtime", return_value=False):
            turn.run()
        assert calls == []

    def test_a_restored_turn_compacts_on_the_primary(self, hooks):
        hooks["route_custody"] = [_refuse_tool_data([])]
        turn, calls = _Turn(), []
        with self._compress_spy(turn, calls):
            turn.run()
        assert calls and set(calls) == {turn.primary}


class TestCheckRequestRoute:
    def _agent(self, *, active=True):
        agent = SimpleNamespace(
            session_id="s1", provider="zai", model="glm-4.7", platform="cli", _provider_fallback_active=active,
            _route_fallback_entry=HOSTED if active else None, _route_turn={"messages": [], "data_use": frozenset()},
            _rate_limited_until=0, _restore_primary_runtime=lambda: False)
        return agent

    def test_every_check_asks_though_nothing_grew(self, hooks):
        asked = []
        hooks["route_custody"] = [_refuse_tool_data(asked, refuse=False)]
        agent = self._agent()
        assert route_hooks.check_request_route(agent, []) is False
        assert route_hooks.check_request_route(agent, []) is False
        assert [phase for phase, *_rest in asked] == ["request", "request"]

    def test_a_gate_that_cannot_run_refuses(self, hooks):
        def broken(**_kw):
            raise RuntimeError("gate down")
        hooks["route_custody"] = [broken]
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.check_request_route(self._agent(), [])

    def test_a_replaced_transcript_is_what_a_failover_reads(self, hooks):
        hooks["route_custody"] = [_refuse_tool_data([], refuse=False)]
        agent, row = self._agent(), {"role": "tool", "content": RAW}
        stamp_data_use(row, [TOOL_DATA])
        route_hooks.check_request_route(agent, [row])
        assert agent._route_turn["messages"] == [row]
        assert TOOL_DATA in route_hooks.session_data_use("s1")

    def test_the_primary_route_is_judged_too(self, hooks):
        asked = []
        hooks["route_custody"] = [_refuse_tool_data(asked)]
        row = {"role": "tool", "content": RAW}
        stamp_data_use(row, [TOOL_DATA])
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.check_request_route(self._agent(active=False), [row])
        assert asked == [("request", "zai", [TOOL_DATA])]

    def test_the_phase_is_advertised(self):
        assert "request" in route_hooks.HERMES_ROUTE_CUSTODY_PHASES
