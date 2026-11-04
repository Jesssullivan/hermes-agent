"""Route custody for delegated children and the end of an auxiliary walk (agent/route_hooks.py).

A review of TIN-7765 named these open:

1: a delegated child copied a gated route with no record, so its turn start asked nothing.
2: a child's connector data stopped at the child, while its summary reached the parent.
3: ``_try_main_agent_model_fallback`` sent to the main model with no gate call.
4: a gate refused a candidate and built-in discovery then ran with no gate call.
5: ``_resolution_fallback_entry`` read a same-provider model change as a move off the entry.
6: the time-of-check race on copied routes is listed under Not covered in hooks.md.

Each probe drives the production code with stub agents and patched resolvers. No provider is
called."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux
from agent import route_hooks
from agent.message_metadata import stamp_data_use
from hermes_cli.plugins import PluginManager
from tools import delegate_tool
from tools.connectors.data_use import collect_data_use

HOSTED = {"provider": "openrouter", "model": "gpt-4o"}
REQUIRED = {**HOSTED, "route_custody": "required"}
RECORD = route_hooks.resolution_record(REQUIRED, {"provider": "openrouter"}, model="gpt-4o")
TOOL_DATA = "acme:client-raw"
HOOKS_DOC = Path(__file__).parents[2] / "website/docs/user-guide/features/hooks.md"


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture(autouse=True)
def _fresh_route_context():
    token = route_hooks._ROUTE_SCOPE.set(None)
    with route_hooks._SESSIONS_LOCK:
        saved = dict(route_hooks._SESSIONS)
        route_hooks._SESSIONS.clear()
    yield
    with route_hooks._SESSIONS_LOCK:
        route_hooks._SESSIONS.clear()
        route_hooks._SESSIONS.update(saved)
    route_hooks._ROUTE_SCOPE.reset(token)


def _agent(provider="openrouter", model="gpt-4o", session_id="s-parent"):
    from tests.agent.test_route_hooks import _make_agent
    agent = _make_agent(None)
    agent.provider, agent.model, agent.session_id = provider, model, session_id
    return agent


def _parent_on_entry():
    parent = _agent()
    route_hooks.bind_resolution_entry(parent, dict(RECORD))
    return parent


def _skip_all(asked):
    def custody(phase, candidate, data_use, labels, task="", **_kw):
        asked.append((phase, candidate["provider"], candidate["model"], sorted(data_use), task))
        return {"action": "skip", "reason": "held"}
    return custody


def _child_built_by_delegation(parent, *, provider="openrouter", model="gpt-4o"):
    """``_build_child_agent`` end to end, with the agent class and config replaced."""
    child = _agent(provider=provider, model=model, session_id="s-child")
    parent.base_url, parent._client_kwargs = "https://example.invalid/v1", {}
    parent.enabled_toolsets, parent.valid_tool_names, parent._delegate_depth = ["terminal"], set(), 0
    parent.tool_progress_callback, parent.prefill_messages = None, None
    runtime = {"provider": provider, "model": model, "base_url": parent.base_url}
    with patch("run_agent.AIAgent", return_value=child), \
            patch.object(delegate_tool, "_resolve_child_runtime", return_value=runtime), \
            patch.object(delegate_tool, "_load_config", return_value={}), \
            patch.object(delegate_tool, "_open_child_session_db", return_value=None), \
            patch.object(delegate_tool, "_resolve_child_credential_pool", return_value=None), \
            patch.object(delegate_tool, "_attach_child"):
        built = delegate_tool._build_child_agent(
            task_index=0, goal="g", context=None, toolsets=None, model=None, max_iterations=1, task_count=1,
            parent_agent=parent)
    assert built is child
    return child


class TestDelegatedChildRoute:
    def test_a_child_on_a_gated_entry_is_checked_at_its_turn_start(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        child = _child_built_by_delegation(_parent_on_entry())
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(child, messages=[{"role": "user", "content": "g"}], turn_id="t1")
        assert [a[:3] for a in asked] == [("turn", "openrouter", "gpt-4o")]

    def test_a_child_of_a_kept_fallback_is_checked(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        parent = _agent()
        parent._provider_fallback_active, parent._route_fallback_entry = True, dict(REQUIRED)
        child = _child_built_by_delegation(parent)
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(child, messages=[{"role": "user", "content": "g"}], turn_id="t1")
        assert [a[0] for a in asked] == ["turn"]

    def test_a_required_entry_with_no_gate_stops_the_child(self, hooks):
        child = _child_built_by_delegation(_parent_on_entry())
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(child, messages=[], turn_id="t1")

    def test_the_child_reads_the_parents_data_use_and_labels(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        parent = _parent_on_entry()
        route_hooks._grow_session(parent, labels=frozenset({"acme:client"}), data_use=frozenset({TOOL_DATA}))
        child = _child_built_by_delegation(parent)
        assert route_hooks._session_labels(child) == {"acme:client"}
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(child, messages=[], turn_id="t1")
        assert asked[0][3] == [TOOL_DATA]

    def test_delegation_model_on_the_parents_provider_is_judged_on_the_childs_model(self, hooks):
        asked = []

        def custody(phase, candidate, **_kw):
            asked.append((phase, candidate["provider"], candidate["model"]))
            return {"action": "skip"} if candidate["model"] == "gpt-5" else {"action": "allow"}

        hooks["route_custody"] = [custody]
        child = _child_built_by_delegation(_parent_on_entry(), model="gpt-5")
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(child, messages=[{"role": "user", "content": "g"}], turn_id="t1")
        assert asked == [("turn", "openrouter", "gpt-5")]

    def test_an_override_to_another_provider_carries_no_entry_and_keeps_the_data(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "skip"}]
        parent = _parent_on_entry()
        route_hooks._grow_session(parent, data_use=frozenset({TOOL_DATA}))
        child = _child_built_by_delegation(parent, provider="anthropic", model="claude-opus-4")
        assert getattr(child, "_route_resolution_entry", None) is None
        assert route_hooks._session_data_use(child) == {TOOL_DATA}
        with pytest.raises(route_hooks.RouteCustodyRefused, match="anthropic/claude-opus-4"):
            route_hooks.begin_turn(child, messages=[], turn_id="t1")  # judged as its own serving route

    def test_an_ungated_build_changes_nothing(self, hooks):
        child = _child_built_by_delegation(_agent())
        assert getattr(child, "_route_resolution_entry", None) is None
        assert route_hooks.session_data_use("s-child") == frozenset()


def _stamped_rows():
    row = {"role": "tool", "tool_call_id": "c1", "content": "raw rows"}
    stamp_data_use(row, [TOOL_DATA])
    return [{"role": "user", "content": "g"}, row, {"role": "assistant", "content": "summary"}]


def _run_child(parent, child):
    entry = {"task_index": 0, "status": "completed"}
    run = MagicMock()
    run.await_child.return_value = (child.result, None, False)
    run.attach_worktree.return_value = entry
    with patch.object(delegate_tool, "_ChildRun", return_value=run), \
            patch.object(delegate_tool, "_lease_child_credential", return_value=(None, None)), \
            patch.object(delegate_tool, "_start_heartbeat", return_value=MagicMock()), \
            patch.object(delegate_tool, "_register_child", return_value=None), \
            patch.object(delegate_tool, "_validate_child_output_schema", return_value=None), \
            patch.object(delegate_tool, "_merge_late_steer"), \
            patch.object(delegate_tool, "_build_result_entry", return_value=entry):
        return delegate_tool._run_single_child(0, "g", child, parent)


class TestChildDataUseReachesParent:
    def test_the_tool_row_sink_and_the_session_take_the_childs_data_use(self, hooks):
        parent = _agent()
        child = _agent(session_id="s-child")
        child.result = {"messages": _stamped_rows(), "final_response": "summary"}
        with collect_data_use() as sink:
            _run_child(parent, child)
        assert sink == {TOOL_DATA}
        assert route_hooks.session_data_use("s-parent") == {TOOL_DATA}

    def test_a_child_that_ends_with_no_result_still_carries_its_rows(self, hooks):
        parent = _agent()
        child = _agent(session_id="s-child")
        child._session_messages, child.result = _stamped_rows(), None
        with collect_data_use() as sink:
            _run_child(parent, child)
        assert sink == {TOOL_DATA}
        assert route_hooks.session_data_use("s-parent") == {TOOL_DATA}

    def test_a_child_that_raised_still_carries_its_rows(self, hooks):
        parent = _agent()
        child = _agent(session_id="s-child")
        child._session_messages = _stamped_rows()
        run = MagicMock()
        run.await_child.side_effect = RuntimeError("boom")
        with patch.object(delegate_tool, "_ChildRun", return_value=run), \
                patch.object(delegate_tool, "_lease_child_credential", return_value=(None, None)), \
                patch.object(delegate_tool, "_start_heartbeat", return_value=MagicMock()), \
                patch.object(delegate_tool, "_register_child", return_value=None), \
                patch.object(delegate_tool, "_fabricated_entry", return_value={}):
            delegate_tool._run_single_child(0, "g", child, parent)
        assert route_hooks.session_data_use("s-parent") == {TOOL_DATA}

    def test_a_child_compaction_keeps_the_data_it_held(self, hooks):
        parent = _agent()
        child = _agent(session_id="s-child")
        route_hooks._grow_session(child, data_use=frozenset({TOOL_DATA}))
        route_hooks.carry_child_data_use(parent, child, {"messages": [{"role": "assistant", "content": "summary"}]})
        assert route_hooks.session_data_use("s-parent") == {TOOL_DATA}

    def test_the_parent_is_asked_again_before_its_next_request(self, hooks):
        asked = []
        hooks["route_custody"] = [lambda phase, data_use, **_kw:
                                  asked.append((phase, sorted(data_use))) or {"action": "allow"}]
        parent = _parent_on_entry()
        route_hooks.begin_turn(parent, messages=[], turn_id="t1")
        assert asked == [("turn", [])]
        child = _agent(session_id="s-child")
        route_hooks.carry_child_data_use(parent, child, {"messages": _stamped_rows()})
        assert route_hooks.check_request_route(parent, []) is False
        assert asked[-1] == ("request", [TOOL_DATA])

    def test_a_clean_child_leaves_the_parent_alone(self, hooks):
        parent = _agent()
        route_hooks.carry_child_data_use(parent, _agent(session_id="s-child"), {"messages": []})
        assert route_hooks.session_data_use("s-parent") == frozenset()


def _main_route(provider="openrouter", model="gpt-4o"):
    return patch.multiple(aux, _read_main_provider=lambda: provider, _read_main_model=lambda: model,
                          _custom_health_base_url=lambda *_a: "", _is_provider_unhealthy=lambda *_a: False,
                          main_model_fallback_refused=lambda *_a: False)


class TestMainModelFallback:
    def test_a_refusal_keeps_the_call_off_the_main_model(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        with _main_route(), patch.object(aux, "resolve_provider_client") as resolve, \
                route_hooks.auxiliary_request([]):
            assert aux._try_main_agent_model_fallback("zai", "compression") == (None, None, "")
        resolve.assert_not_called()
        assert asked == [("auxiliary", "openrouter", "gpt-4o", [], "compression")]

    def test_an_allow_reaches_the_main_model(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        client = MagicMock()
        with _main_route(), patch.object(aux, "resolve_provider_client", return_value=(client, "gpt-4o")):
            assert aux._try_main_agent_model_fallback("zai", "compression")[0] is client

    def test_a_required_main_route_with_no_gate_is_refused(self, hooks):
        agent = _parent_on_entry()
        route_hooks._bind_scope(agent)
        with _main_route(), patch.object(aux, "resolve_provider_client") as resolve:
            assert aux._try_main_agent_model_fallback("zai", "compression") == (None, None, "")
        resolve.assert_not_called()

    def test_the_gate_sees_the_turns_data_use(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        agent = _agent()
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=_stamped_rows(), turn_id="t1")  # binds the scope all the same
        with _main_route(), patch.object(aux, "resolve_provider_client"):
            aux._try_main_agent_model_fallback("zai", "compression")
        assert asked[-1][0] == "auxiliary" and asked[-1][3] == [TOOL_DATA]

    def test_with_no_gate_nothing_changes(self, hooks):
        client = MagicMock()
        with _main_route(), patch.object(aux, "resolve_provider_client", return_value=(client, "gpt-4o")):
            assert aux._try_main_agent_model_fallback("zai", "compression")[0] is client


class TestDiscoveryStops:
    def _provider_chain(self):
        tried = MagicMock(return_value=(MagicMock(), "any-model"))
        return tried, patch.object(aux, "_get_provider_chain", lambda: [("openrouter", tried)])

    def _refuse_one(self):
        assert route_hooks.auxiliary_refuses(HOSTED, task="compression", failed_provider="zai") is True

    def test_a_refusal_stops_the_payment_chain(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "skip"}]
        tried, chain = self._provider_chain()
        with chain, patch.object(aux, "_read_main_provider", lambda: ""), route_hooks.auxiliary_request([]):
            self._refuse_one()
            assert aux._try_payment_fallback("zai", "compression") == (None, None, "")
        tried.assert_not_called()

    def test_a_required_entry_with_no_gate_counts_as_a_refusal(self, hooks):
        tried, chain = self._provider_chain()
        with chain, patch.object(aux, "_read_main_provider", lambda: ""), route_hooks.auxiliary_request([]):
            assert route_hooks.auxiliary_refuses(REQUIRED, task="compression", failed_provider="zai") is True
            assert aux._try_payment_fallback("zai", "compression") == (None, None, "")
        tried.assert_not_called()

    def test_the_walk_before_a_refusal_keeps_discovery(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        tried, chain = self._provider_chain()
        with chain, patch.object(aux, "_read_main_provider", lambda: ""), \
                patch.object(aux, "_custom_health_base_url", lambda *_a: ""), route_hooks.auxiliary_request([]):
            assert route_hooks.auxiliary_refuses(HOSTED, task="compression", failed_provider="zai") is False
            assert aux._try_payment_fallback("zai", "compression")[0] is not None
        tried.assert_called_once()

    def test_a_refusal_ends_with_its_own_call(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "skip"}]
        with route_hooks.auxiliary_request([]):
            self._refuse_one()
            assert route_hooks.discovery_refused("compression") is True
        with route_hooks.auxiliary_request([]):
            assert route_hooks.discovery_refused("compression") is False

    def test_the_auto_route_outside_a_request_stops_too(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "skip"}]
        chain = [dict(HOSTED)]
        with patch.object(aux, "_try_main_provider_route", return_value=None), \
                patch.object(aux, "_main_route_target", return_value=("", "", "", "", "")), \
                patch.object(aux, "_try_configured_fallback_chain", return_value=(None, None, "")), \
                patch("hermes_cli.config.load_config_readonly", return_value={}), \
                patch("hermes_cli.fallback_config.get_fallback_chain", return_value=chain), \
                patch.object(aux, "_custom_health_base_url", lambda *_a: ""), \
                patch.object(aux, "_try_discovery_chain") as discovery:
            assert aux._resolve_auto_route(task="compression") == (None, None, "")
        discovery.assert_not_called()

    def test_the_auto_route_with_no_refusal_keeps_discovery(self, hooks):
        with patch.object(aux, "_try_main_provider_route", return_value=None), \
                patch.object(aux, "_main_route_target", return_value=("", "", "", "", "")), \
                patch.object(aux, "_try_configured_fallback_chain", return_value=(None, None, "")), \
                patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")), \
                patch.object(aux, "_try_discovery_chain", return_value=("client", "m", "openrouter")):
            assert aux._resolve_auto_route(task="compression")[0] == "client"

    def test_the_ladder_stops_after_a_refused_chain(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "skip"}]
        tried, chain = self._provider_chain()
        route = SimpleNamespace(base_info="", main_runtime=None)
        with chain, patch.object(aux, "_read_main_provider", lambda: ""), route_hooks.auxiliary_request([]), \
                patch.object(aux, "_try_configured_fallback_chain", side_effect=lambda *_a, **_k: (
                    self._refuse_one() or (None, None, ""))), \
                patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")):
            fb = aux._next_fallback_after_quarantine("compression", "auto", True, route, None, None)
        assert fb == (None, None, "")
        tried.assert_not_called()


class TestSameProviderModelChange:
    def test_a_same_provider_model_change_is_still_checked(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        agent = _parent_on_entry()
        agent.model = "gpt-5"  # /model on the entry's provider
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t1")
        assert [a[:3] for a in asked] == [("turn", "openrouter", "gpt-5")]

    def test_a_required_entry_with_no_gate_refuses_the_new_model(self, hooks):
        agent = _parent_on_entry()
        agent.model = "gpt-5"
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t1")

    def test_a_provider_alias_is_the_same_provider(self, hooks):
        entry = route_hooks.resolution_record(
            {"provider": "claude", "model": "claude-opus-4", "route_custody": "required"}, {"provider": "anthropic"})
        agent = _agent(provider="anthropic", model="claude-opus-4")
        route_hooks.bind_resolution_entry(agent, entry)
        agent.model = "claude-sonnet-4"
        assert route_hooks._resolution_fallback_entry(agent)["model"] == "claude-sonnet-4"

    def test_an_admission_of_the_old_model_does_not_cover_the_new_one(self, hooks):
        asked = []
        hooks["route_custody"] = [lambda phase, candidate, **_kw:
                                  asked.append((phase, candidate["model"])) or {"action": "allow"}]
        agent = _parent_on_entry()
        route_hooks.begin_turn(agent, messages=[], turn_id="t1")
        agent.model = "gpt-5"
        assert route_hooks.check_request_route(agent, []) is False
        assert asked == [("turn", "gpt-4o"), ("request", "gpt-5")]

    def test_the_record_keeps_its_own_model(self, hooks):
        agent = _parent_on_entry()
        agent.model = "gpt-5"
        route_hooks._resolution_fallback_entry(agent)
        assert agent._route_resolution_entry["model"] == "gpt-4o"

    def test_a_blank_provider_proves_no_move(self, hooks):
        agent = _parent_on_entry()
        agent.provider = ""
        assert route_hooks._resolution_fallback_entry(agent) is not None

    def test_another_provider_still_leaves_the_entry(self, hooks):
        agent = _parent_on_entry()
        agent.provider, agent.model = "anthropic", "claude-opus-4"
        assert route_hooks._resolution_fallback_entry(agent) is None


class TestNotCovered:
    def test_the_copied_route_race_is_listed_with_its_reason(self):
        text = HOOKS_DOC.read_text(encoding="utf-8")
        listed = text.split("**Not covered:**", 1)[1].split("### ", 1)[0]
        assert "time-of-check race on copied routes" in listed
        assert "No lock spans the two" in listed
