"""F12: custody on the serving route (agent/route_hooks.py ``check_request_route``).

The serving route is judged before every API request of a turn, whatever route it is, and
``model.route_custody: required`` fails closed. A delegated child on ``delegation.model`` and a
gateway ``/model`` on a resolution-time entry are judged on the model in use. Each probe drives the
production code with a stub provider call. No provider is called."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux
from agent import route_hooks
from agent.message_metadata import stamp_data_use
from hermes_cli.config import get_config_path
from hermes_cli.config_read_errors import FailedConfigRead
from hermes_cli.plugins import PluginManager
from tests.agent.test_route_hooks_turn_requests import HOSTED, RAW, TOOL_DATA, _Turn

LOCAL = {"provider": "openai", "model": "gpt-4o", "api_key": "local-secret"}
RECORD = {"provider": "zai", "model": "glm-4.7", "resolved_provider": "zai", "resolved_model": "glm-4.7",
          "route_custody": "required"}


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


@pytest.fixture
def marked():
    """``model.route_custody: required`` on the ``_Turn`` primary, without a config file."""
    with patch("agent.route_hooks._main_route_marker", return_value=("openrouter", True)):
        yield


def _gate(asked, refuse_on=(), *, data_only=True):
    """Allows, except the tool data (or anything, without *data_only*) on the routes in *refuse_on*."""
    def custody(phase, candidate, data_use, **_kw):
        asked.append((phase, candidate["provider"], candidate["model"], list(data_use)))
        if candidate["provider"] in refuse_on and (TOOL_DATA in data_use or not data_only):
            return {"action": "skip", "reason": "client data"}
        return {"action": "allow"}
    return custody


class TestMarkedMainRoute:
    def test_no_gate_sends_nothing(self, hooks, marked):
        turn = _Turn(chain=(), fail_primary=False)
        result = turn.run()
        assert turn.sent == [] and result["failure_reason"] == "route_custody" and result["api_calls"] == 0

    def test_no_gate_moves_to_the_first_route_that_needs_none(self, hooks, marked):
        turn = _Turn(chain=(HOSTED,), fail_primary=False)
        result = turn.run()
        assert turn.sent and {provider for provider, _raw in turn.sent} == {"zai"}
        assert result["final_response"] == "done"

    def test_no_allow_sends_nothing(self, hooks, marked):
        hooks["route_custody"] = [lambda **_kw: None]
        turn = _Turn(chain=(), fail_primary=False)
        assert turn.run()["failure_reason"] == "route_custody" and turn.sent == []

    def test_a_gate_error_sends_nothing(self, hooks, marked):
        def broken(**_kw):
            raise RuntimeError("gate down")
        hooks["route_custody"] = [broken]
        turn = _Turn(chain=(), fail_primary=False)
        assert turn.run()["failure_reason"] == "route_custody" and turn.sent == []

    def test_a_gate_timeout_sends_nothing(self, hooks, marked, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1)
        release = threading.Event()
        hooks["route_custody"] = [lambda **_kw: release.wait(timeout=10.0) and None]
        turn = _Turn(chain=(), fail_primary=False)
        try:
            assert turn.run()["failure_reason"] == "route_custody" and turn.sent == []
        finally:
            release.set()

    def test_an_allow_carries_the_turn(self, hooks, marked):
        hooks["route_custody"] = [_gate([])]
        turn = _Turn(chain=(), fail_primary=False)
        assert turn.run()["final_response"] == "done" and turn.sent[-1] == (turn.primary, True)

    def test_a_failed_labeller_refuses_a_marked_route(self, hooks, marked):
        def labeller(**_kw):
            raise RuntimeError("classifier down")
        hooks["turn_route"], hooks["route_custody"] = [labeller], [_gate([])]
        turn = _Turn(chain=(), fail_primary=False)
        assert turn.run()["failure_reason"] == "route_custody" and turn.sent == []

    def test_an_unmarked_main_route_stays_through_a_failed_labeller(self, hooks):
        def labeller(**_kw):
            raise RuntimeError("classifier down")
        hooks["turn_route"], hooks["route_custody"] = [labeller], [_gate([])]
        turn = _Turn(chain=(), fail_primary=False)
        assert turn.run()["final_response"] == "done" and turn.sent

    def test_the_marker_names_its_own_provider(self, hooks):
        with patch("agent.route_hooks._main_route_marker", return_value=("zai", True)):
            turn = _Turn(chain=(), fail_primary=False)
            assert turn.run()["final_response"] == "done"

    def test_outside_a_turn_a_marked_route_refuses_with_no_gate(self, hooks, marked):
        agent = _Turn(chain=(), fail_primary=False).agent
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.check_out_of_turn_route(agent, phase="compaction", inbound=[{"role": "user", "content": "q"}])

    def test_an_auto_slot_does_not_ride_a_refused_main_route(self, hooks, marked):
        route_hooks._ROUTE_SCOPE.set(None)
        with patch.object(aux, "_main_route_target", return_value=("openrouter", "gpt-4o", "", "", "")), \
                patch.object(aux, "_try_main_provider_route") as ride, \
                patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")), \
                patch.object(aux, "_try_discovery_chain") as discovery, \
                patch.object(aux, "_warn_stale_openai_base_url"), \
                patch.object(aux, "_normalize_main_runtime", return_value={"provider": "openrouter"}):
            assert aux._resolve_auto_route(task=None)[0] is None
        ride.assert_not_called()
        discovery.assert_not_called()

    def test_an_auto_vision_slot_sends_no_image_on_a_refused_main_route(self, hooks, marked):
        route_hooks._ROUTE_SCOPE.set(None)
        with patch.object(aux, "_vision_main_provider_client") as ride, \
                patch.object(aux, "_resolve_strict_vision_backend") as discovery:
            assert aux._vision_auto_route({"provider": "openrouter", "model": "m"}, None, None, False) == (None, None, None)
        ride.assert_not_called()
        discovery.assert_not_called()

    def test_an_auto_vision_slot_rides_an_admitted_main_route(self, hooks, marked):
        route_hooks._ROUTE_SCOPE.set(None)
        asked = []
        hooks["route_custody"] = [lambda phase, task, **_kw: asked.append((phase, task)) or {"action": "allow"}]
        with patch.object(aux, "_vision_main_provider_client", return_value=(None, None)) as ride, \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(None, None)):
            aux._vision_auto_route({"provider": "openrouter", "model": "m"}, None, None, False)
        ride.assert_called_once()
        assert asked == [("auxiliary", "vision")]

    def test_a_vision_slot_naming_the_marked_route_gets_no_client(self, hooks, marked):
        route_hooks._ROUTE_SCOPE.set(None)
        with patch.object(aux, "_resolve_task_provider_model", return_value=("openrouter", "m", None, None, None)), \
                patch.object(aux, "_resolve_strict_vision_backend") as strict:
            assert aux.resolve_vision_provider_client()[1] is None
        strict.assert_not_called()

    def test_a_vision_availability_probe_never_asks_the_gate(self, hooks, marked):
        route_hooks._ROUTE_SCOPE.set(None)
        asked = []
        hooks["route_custody"] = [lambda **_kw: asked.append(1) or {"action": "skip"}]
        with aux.aux_probe_mode(), patch.object(aux, "_vision_main_provider_client", return_value=(None, None)) as ride, \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(None, None)):
            aux._vision_auto_route({"provider": "openrouter", "model": "m"}, None, None, False)
        ride.assert_called_once()
        assert asked == []

    def test_a_slot_naming_the_marked_route_gets_no_client(self, hooks, marked):
        with patch.object(aux, "resolve_provider_client") as build:
            assert aux._get_cached_client("openrouter", "gpt-4o", task="vision") == (None, None)
        build.assert_not_called()

    def test_an_auto_client_on_a_marked_route_is_judged_each_call(self, hooks, marked):
        answers = iter(({"action": "allow"}, {"action": "skip"}))
        hooks["route_custody"] = [lambda **_kw: next(answers)]
        main = {"provider": "openrouter", "model": "gpt-4o", "base_url": "https://u/v1", "api_key": "k", "api_mode": ""}
        client = MagicMock()
        with patch.object(aux, "_try_main_provider_route", return_value=(client, "gpt-4o", "openrouter")) as ride, \
                patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")), \
                patch.object(aux, "_warn_stale_openai_base_url"), \
                patch.object(aux, "_read_main_provider", return_value="openrouter"), \
                patch.object(aux, "_read_main_model", return_value="gpt-4o"):
            assert aux._get_cached_client("auto", main_runtime=main)[0] is client
            assert aux._get_cached_client("auto", main_runtime=main)[0] is None, "the cached client was served"
        ride.assert_called_once()

    def test_an_auto_slot_rides_an_admitted_main_route(self, hooks, marked):
        hooks["route_custody"] = [_gate([])]
        client = MagicMock()
        with patch.object(aux, "_main_route_target", return_value=("openrouter", "gpt-4o", "", "", "")), \
                patch.object(aux, "_try_main_provider_route", return_value=(client, "gpt-4o", "openrouter")), \
                patch.object(aux, "_warn_stale_openai_base_url"), \
                patch.object(aux, "_normalize_main_runtime", return_value={"provider": "openrouter"}):
            assert aux._resolve_auto_route(task=None)[0] is client


class TestMarkerRead:
    @pytest.fixture(autouse=True)
    def _clean(self):
        saved = dict(route_hooks._MAIN_MARKERS)
        route_hooks._MAIN_MARKERS.clear()
        yield
        route_hooks._MAIN_MARKERS.clear()
        route_hooks._MAIN_MARKERS.update(saved)

    @staticmethod
    def _write(text):
        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_the_block_marks_its_provider_only(self):
        self._write("model:\n  provider: claude\n  default: m\n  route_custody: required\n")
        assert route_hooks.main_route_marked("anthropic") and route_hooks.main_route_marked("claude")
        assert not route_hooks.main_route_marked("zai")
        assert route_hooks.entry_requires_custody({"provider": "anthropic", "model": "m", "route_custody": "off"})

    @pytest.mark.parametrize("name", ["custom:relay", "relay"])
    def test_a_named_custom_block_marks_the_route_it_runs_as(self, name):
        self._write(f"model:\n  provider: {name}\n  default: m\n  route_custody: required\n"
                    "custom_providers:\n- name: relay\n  base_url: http://relay.invalid/v1\n")
        assert route_hooks.main_route_marked("custom")
        assert not route_hooks.main_route_marked("zai")

    @pytest.mark.parametrize("text,expected", [
        ("model:\n  provider: auto\n  route_custody: required\n", True),
        ("model:\n  route_custody: required\n", True),
        ("model:\n  provider: zai\n  route_custody: yes-please\n", True),
        ("model:\n  provider: zai\n  route_custody: off\n", False),
        ("model:\n  provider: zai\n  default: m\n", False),
        ("model: plain-string\n", False),
        ("{}\n", False),
    ])
    def test_values(self, text, expected):
        self._write(text)
        assert route_hooks.main_route_marked("zai") is expected

    def test_a_failed_read_keeps_the_last_good_answer(self):
        self._write("model:\n  provider: zai\n  route_custody: required\n")
        assert route_hooks.main_route_marked("zai")
        failed = FailedConfigRead(error=OSError("io"))
        with patch("hermes_cli.config.read_raw_config_readonly", return_value=failed):
            assert route_hooks.main_route_marked("zai") and not route_hooks.main_route_marked("openai")
        self._write("model:\n  provider: zai\n")
        assert not route_hooks.main_route_marked("zai")
        with patch("hermes_cli.config.read_raw_config_readonly", return_value=failed):
            assert not route_hooks.main_route_marked("zai")

    @pytest.mark.parametrize("read", [{"return_value": FailedConfigRead(error=OSError("io"))},
                                      {"side_effect": RuntimeError("no config module")}])
    def test_with_no_good_read_the_route_counts_as_marked(self, read):
        with patch("hermes_cli.config.read_raw_config_readonly", **read):
            assert route_hooks.main_route_marked("anything")


class TestEveryRequest:
    def _ordered(self, turn, asked):
        """``turn.send`` beside the gate's log, so a request with no ask before it shows."""
        order, send = [], turn.send

        def logged(api_kwargs, **kw):
            order.append(("send", len(asked)))
            return send(api_kwargs, **kw)

        return order, logged

    def test_the_gate_answers_for_each_request_on_the_primary(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        turn = _Turn(chain=(), fail_primary=False)
        order, turn.send = self._ordered(turn, asked)
        turn.run()
        assert [seen for _kind, seen in order] == sorted({seen for _kind, seen in order}), "a request went unasked"
        assert len(order) == 2 and asked[0][0] == "turn"
        assert {a[1] for a in asked} == {"openrouter"} and asked[-1][3] == [TOOL_DATA]

    def test_each_request_on_a_kept_fallback_is_asked_though_nothing_grew(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        turn = _Turn()
        order, turn.send = self._ordered(turn, asked)
        turn.run()
        assert len(order) == 3 and order[-1][1] > order[-2][1]
        assert next(a[0] for a in asked if a[1] == "zai") == "failover"

    def test_a_refused_row_moves_off_the_primary(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, {"openrouter"})]
        turn = _Turn(fail_primary=False)
        result = turn.run()
        assert (turn.primary, True) not in turn.sent and turn.sent[-1] == ("zai", True)
        assert result["final_response"] == "done" and turn.agent._provider_fallback_active

    def test_a_refused_row_with_no_route_left_stops_with_f6(self, hooks):
        fired = []
        hooks["route_custody"] = [_gate([], {"openrouter", "zai"})]
        hooks["on_routes_exhausted"] = [lambda reason, **_kw: fired.append(reason) or {"reply": "Held."}]
        turn = _Turn(fail_primary=False)
        result = turn.run()
        assert not any(raw for _provider, raw in turn.sent), "the row never left"
        assert fired == ["route_custody"] and result["final_response"] == "Held."
        assert result["failure_reason"] == "route_custody"
        assert any(RAW in str(m.get("content")) for m in result["messages"]), "the row stays in the transcript"

    def test_a_refused_kept_fallback_and_primary_reach_the_next_entry(self, hooks):
        hooks["route_custody"] = [_gate([], {"openrouter", "zai"})]
        turn = _Turn(chain=(HOSTED, LOCAL))
        result = turn.run()
        assert turn.sent[-1] == ("openai", True) and result["final_response"] == "done"
        assert not any(raw for provider, raw in turn.sent if provider != "openai")

    def test_a_retry_is_a_request(self, hooks):
        asked = []

        def custody(phase, candidate, **_kw):
            asked.append(phase)
            return {"action": "skip"} if len(asked) >= 3 else {"action": "allow"}

        hooks["route_custody"] = [custody]
        turn = _Turn(chain=(), fail_primary=False)
        send = turn.send

        def flaky(api_kwargs, **kw):
            send(api_kwargs, **kw)
            error = Exception("upstream down")
            error.status_code = 500
            raise error

        turn.send = flaky
        result = turn.run()
        assert asked == ["turn", "request", "request"] and len(turn.sent) == 1
        assert result["failure_reason"] == "route_custody" and result["api_calls"] == 1

    def test_a_failover_inside_the_retry_loop_is_judged_before_it_sends(self, hooks):
        hooks["route_custody"] = [_gate([], {"zai"}, data_only=False)]
        turn = _Turn(chain=(HOSTED,))
        result = turn.run()
        assert {provider for provider, _raw in turn.sent} == {turn.primary}, "the refused entry sent nothing"
        assert result.get("failed")

    def test_the_budget_summary_is_a_request(self, hooks):
        hooks["route_custody"] = [_gate([], {"openrouter"})]
        turn = _Turn(chain=(), fail_primary=False)
        turn.agent.max_iterations = 1
        result = turn.run()
        assert turn.sent == [(turn.primary, False)], "the summary never left"
        assert result["failure_reason"] == "route_custody"
        assert not any("summary" in str(m.get("content")).lower() and m.get("role") == "user" for m in result["messages"])

    def test_the_budget_summary_moves_with_a_refusal(self, hooks):
        hooks["route_custody"] = [_gate([], {"openrouter"})]
        turn = _Turn(chain=(HOSTED,), fail_primary=False)
        turn.agent.max_iterations = 1
        turn.run()
        assert turn.sent[-1] == ("zai", True)

    def test_labels_grown_in_the_turn_reach_the_gate(self, hooks):
        phases, seen = [], []

        def labeller(phase, data_use, **_kw):
            phases.append(phase)
            return {"labels": ["acme:client"]} if TOOL_DATA in data_use else None

        def custody(labels, **_kw):
            seen.append(list(labels))
            return {"action": "allow"}

        hooks["turn_route"], hooks["route_custody"] = [labeller], [custody]
        turn = _Turn(chain=(), fail_primary=False)
        turn.run()
        assert phases[0] == "turn" and set(phases[1:]) == {"request"}
        assert seen[0] == [] and seen[-1] == ["acme:client"]

    def test_a_labeller_failing_mid_turn_stops_a_marked_route(self, hooks, marked):
        def labeller(phase, data_use, **_kw):
            if TOOL_DATA in data_use:
                raise RuntimeError("classifier down")

        hooks["turn_route"], hooks["route_custody"] = [labeller], [_gate([])]
        turn = _Turn(chain=(), fail_primary=False)
        assert turn.run()["failure_reason"] == "route_custody"
        assert turn.sent == [(turn.primary, False)]

    def test_a_check_that_cannot_finish_stops_the_turn(self, hooks):
        hooks["route_custody"] = [_gate([])]
        turn = _Turn(chain=(), fail_primary=False)
        with patch("agent.turn_route_check.check_request_route", side_effect=RuntimeError("bug")):
            assert turn.run()["failure_reason"] == "route_custody" and turn.sent == []

    def test_no_hook_and_no_marker_asks_nothing(self, hooks):
        turn = _Turn(chain=(), fail_primary=False)
        with patch("agent.route_hooks.fallback_custody_refuses") as asked:
            assert turn.run()["final_response"] == "done"
        asked.assert_not_called()

    def test_a_restored_primary_answers_before_the_row_leaves(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, {"openrouter", "zai"})]
        agent = _agent()
        agent._provider_fallback_active, agent._route_fallback_entry = True, HOSTED
        agent._route_turn = {"messages": [], "data_use": frozenset()}

        def restore():
            agent.provider, agent.model, agent._provider_fallback_active = "openrouter", "gpt-4o", False
            return True

        agent._restore_primary_runtime = restore
        row = {"role": "tool", "content": RAW}
        stamp_data_use(row, [TOOL_DATA])
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.check_request_route(agent, [row])
        assert [(a[0], a[1]) for a in asked] == [("request", "zai"), ("request", "openrouter")]

    def test_a_walk_with_no_route_left_leaves_the_primary_cooldown_alone(self, hooks):
        hooks["route_custody"] = [_gate([], {"openrouter", "zai"})]
        turn = _Turn(chain=(HOSTED,), fail_primary=False)
        turn.agent._rate_limited_until = 0
        assert turn.run()["failure_reason"] == "route_custody"
        assert turn.agent._rate_limited_until == 0

    def test_a_move_at_turn_start_syncs_the_turns_prompt(self, hooks):
        from agent import conversation_loop
        hooks["route_custody"] = [_gate([], {"openrouter"}, data_only=False)]
        turn = _Turn(chain=(HOSTED,), fail_primary=False)
        with patch.object(conversation_loop, "_sync_failover_system_message",
                          wraps=conversation_loop._sync_failover_system_message) as sync:
            turn.run()
        sync.assert_called()
        assert {provider for provider, _raw in turn.sent} == {"zai"}

    def test_a_rate_limit_keeps_the_order(self, hooks):
        """The primary's own 429 still walks the chain, and the custody check rides along."""
        hooks["route_custody"] = [_gate([])]
        turn = _Turn(chain=(HOSTED,))
        assert turn.run()["final_response"] == "done" and turn.sent[-1] == ("zai", True)


def _agent(provider="zai", model="glm-4.7", session_id="s1"):
    from types import SimpleNamespace
    return SimpleNamespace(session_id=session_id, provider=provider, model=model, platform="cli",
                           _provider_fallback_active=False, _fallback_chain=[], _rate_limited_until=0)


class TestModelInUse:
    def test_any_other_model_on_the_entry_is_judged_as_itself(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        agent = _agent(model="glm-5")
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        route_hooks.begin_turn(agent, messages=[], turn_id="t1")
        assert asked[-1][:3] == ("turn", "zai", "glm-5")

    def test_the_entrys_own_model_stays_as_written(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        agent = _agent()
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        route_hooks.begin_turn(agent, messages=[], turn_id="t1")
        assert asked[-1][:3] == ("turn", "zai", "glm-4.7")


class TestGatewayOverrideOnTheEntry:
    """A gateway ``/model`` with a key resolves to ``None``: the session's record stands in."""

    def _after_override(self, agent):
        route_hooks.bind_resolution_entry(agent, None)
        return agent

    def test_a_cached_agent_keeps_its_record(self, hooks):
        agent = _agent()
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        agent.model = "glm-5"
        assert self._after_override(agent)._route_resolution_entry["provider"] == "zai"
        with pytest.raises(route_hooks.RouteCustodyRefused, match="zai/glm-5"):
            route_hooks.begin_turn(agent, messages=[], turn_id="t2")  # required, no gate

    def test_a_rebuilt_agent_takes_the_sessions_record(self, hooks):
        route_hooks.bind_resolution_entry(_agent(), dict(RECORD))
        agent = self._after_override(_agent(model="glm-5"))
        with pytest.raises(route_hooks.RouteCustodyRefused, match="zai/glm-5"):
            route_hooks.begin_turn(agent, messages=[], turn_id="t2")

    def test_the_real_resolver_path_keeps_the_gate_on_the_entry(self, hooks):
        from types import SimpleNamespace
        from tests.gateway.test_pre_agent_fallback_notice import _runner_with_real_runtime_resolution

        asked = []
        hooks["route_custody"] = [_gate(asked, {"zai"}, data_only=False)]
        agent = _agent()
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        runner = _runner_with_real_runtime_resolution()
        override = {"model": "glm-5", "provider": "zai", "api_key": "k", "base_url": "u", "api_mode": "chat_completions"}
        state = SimpleNamespace(conversation=SimpleNamespace(model_override=override))
        runner._peek_session_state = lambda _key: state
        runner._rehydrate_session_model_override = lambda _key: None
        with patch("gateway.run._resolve_gateway_model", return_value="m"):
            model, _runtime = runner._resolve_session_agent_runtime(session_key="s", user_config={})
        assert model == "glm-5"
        agent.model = model
        route_hooks.bind_resolution_entry(agent, route_hooks.take_resolution_entry())
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t2")
        assert asked[-1][:3] == ("turn", "zai", "glm-5")

    def test_the_primary_on_another_provider_clears_it(self, hooks):
        agent = _agent()
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        agent.provider, agent.model = "openrouter", "gpt-4o"
        assert self._after_override(agent)._route_resolution_entry is None
        assert route_hooks._session_entry("s1") is None
        route_hooks.begin_turn(agent, messages=[], turn_id="t2")  # nothing required of it

    def test_a_detached_call_never_forgets_the_sessions_record(self, hooks):
        route_hooks.bind_resolution_entry(_agent(), dict(RECORD))
        with route_hooks.detached_auxiliary(None, task="vision", session_id="s1", provider="openrouter", model="m",
                                            messages=[], rides_main=False):
            pass
        assert route_hooks._session_entry("s1")["provider"] == "zai"

    def test_a_detached_call_on_the_entry_is_checked(self, hooks):
        route_hooks.bind_resolution_entry(_agent(), dict(RECORD))
        with pytest.raises(route_hooks.RouteCustodyRefused), route_hooks.detached_auxiliary(
                None, task="vision", session_id="s1", provider="zai", model="glm-5", messages=[]):
            pass

    def test_a_background_agent_takes_the_sessions_record(self, hooks):
        route_hooks.bind_resolution_entry(_agent(), dict(RECORD))
        child = _agent(model="glm-5", session_id="bg")
        route_hooks.inherit_route_custody(child, None, session_id="s1")
        assert child._route_resolution_entry["provider"] == "zai"

    def test_a_rotated_session_keeps_the_record(self, hooks):
        agent = _agent()
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        agent.session_id = "s2"
        route_hooks.carry_session_route_state(agent, "s1")
        assert route_hooks._session_entry("s2")["provider"] == "zai"
        route_hooks.carry_route_state("s2", "s3")
        assert route_hooks._session_entry("s3")["provider"] == "zai"
