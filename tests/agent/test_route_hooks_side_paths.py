"""Route custody on the CLI walker and on side agents and side questions (agent/route_hooks.py).

CADE-1 review 5728175100 found these open:

B1: the interactive CLI walker (``_resolve_fallback_runtime``) asked no gate, so a ``required``
    entry became the CLI route, and nothing recorded the entry.
B2: gateway ``_side_agent`` (``/bg``, preview restart) and the TUI's side agents copied or resolved
    a gated route with no record, so their turn start asked nothing. ``/btw`` sent the transcript
    on a gated entry with no gate call, and its fork rode a kept fallback.

Each probe drives the production code with stub agents and patched resolvers. No provider is
called."""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import route_hooks
from agent.message_metadata import stamp_data_use
from agent.side_question import answer_side_question
from hermes_cli.auth import AuthError
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.plugins import PluginManager

HOSTED = {"provider": "openai", "model": "gpt-4o"}
REQUIRED = {**HOSTED, "route_custody": "required"}
RECORD = route_hooks.resolution_record(REQUIRED, {"provider": "openrouter"})
TOOL_DATA = "acme:client-raw"


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture(autouse=True)
def _fresh_route_context():
    tokens = [(var, var.set(None)) for var in (route_hooks._ROUTE_SCOPE, route_hooks._RESOLVED_ENTRY)]
    with route_hooks._SESSIONS_LOCK:
        saved = dict(route_hooks._SESSIONS)
        route_hooks._SESSIONS.clear()
    yield
    with route_hooks._SESSIONS_LOCK:
        route_hooks._SESSIONS.clear()
        route_hooks._SESSIONS.update(saved)
    for var, token in reversed(tokens):
        var.reset(token)


def _skip_all(asked):
    def custody(phase, candidate, data_use, labels, task="", **_kw):
        asked.append((phase, candidate["provider"], task, sorted(data_use), sorted(labels)))
        return {"action": "skip", "reason": "held"}
    return custody


def _allow_resolve_only(asked):
    def custody(phase, candidate, data_use, labels, task="", **_kw):
        asked.append((phase, candidate["provider"], task, sorted(data_use), sorted(labels)))
        return {"action": "allow"} if phase == "resolve" else {"action": "skip", "reason": "client data"}
    return custody


def _stamped_history():
    row = {"role": "tool", "tool_call_id": "c1", "content": "raw rows"}
    stamp_data_use(row, [TOOL_DATA])
    return [{"role": "user", "content": "q"}, {"role": "assistant", "tool_calls": []}, row,
            {"role": "assistant", "content": "a"}]


def _stub_agent(provider="openrouter", model="gpt-4o", session_id="side"):
    return SimpleNamespace(provider=provider, model=model, session_id=session_id, platform="tui",
                           _provider_fallback_active=False, _rate_limited_until=0,
                           _restore_primary_runtime=lambda: False)


def _kept_fallback_parent(entry=REQUIRED, session_id="parent"):
    return SimpleNamespace(provider="openai", model="gpt-4o", session_id=session_id, platform="tui",
                           _provider_fallback_active=True, _route_fallback_entry=entry, _fallback_chain=[entry])


# -- B1: the interactive CLI walker -----------------------------------------------------------------


def _shell(chain):
    shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    shell._fallback_model = chain
    shell.requested_provider, shell.model, shell.session_id = "anthropic", "claude-sonnet-4", "cli-session"
    return shell


def _walk(shell):
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"provider": "openrouter", "api_key": "k", "base_url": "u"}), \
            patch("hermes_cli.fallback_config.resolve_entry_api_key", return_value="k"), \
            patch("gateway.warning_notifications.render_notification"):
        return shell._resolve_fallback_runtime(AuthError("no key", provider="anthropic", code="missing_api_key"))


class TestCliWalker:
    def test_a_required_entry_with_no_gate_is_skipped(self, hooks):
        shell = _shell([REQUIRED])
        assert _walk(shell) is None
        assert (shell.model, shell.requested_provider) == ("claude-sonnet-4", "anthropic")
        assert getattr(shell, "_route_entry_record", None) is None

    def test_the_gate_is_asked_and_its_refusal_holds(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        shell = _shell([HOSTED])
        assert _walk(shell) is None
        assert asked == [("resolve", "openai", "", [], [])] and shell.model == "claude-sonnet-4"

    def test_the_next_entry_carries_on_after_a_skip(self, hooks):
        shell = _shell([REQUIRED, {"provider": "zai", "model": "glm-4.7"}])
        assert _walk(shell) is not None
        assert (shell.model, shell.requested_provider) == ("glm-4.7", "zai")

    def test_an_allowed_entry_is_recorded_and_sees_the_session_labels(self, hooks):
        asked = []
        hooks["route_custody"] = [_allow_resolve_only(asked)]
        route_hooks._remember_session("cli-session", labels=frozenset({"acme:client"}))
        shell = _shell([REQUIRED])
        assert _walk(shell) is not None
        assert asked[0][4] == ["acme:client"]
        assert shell._route_entry_record["route_custody"] == "required"
        assert shell._route_entry_record["resolved_provider"] == "openrouter"

    def test_no_gate_and_no_key_walks_as_before(self, hooks):
        shell = _shell([HOSTED])
        with patch("agent.route_hooks._invoke", side_effect=AssertionError("no hook may run")):
            assert _walk(shell) is not None
        assert shell.model == "gpt-4o"

    def test_the_record_reaches_the_agent_the_shell_builds(self, hooks):
        import cli

        assert cli  # the mixin writes this module's active-agent reference

        class Shell(CLIAgentSetupMixin):
            def __getattr__(self, name):
                return MagicMock()

        shell = Shell.__new__(Shell)
        shell._route_entry_record = dict(RECORD)
        shell._ensure_runtime_credentials = lambda: True
        shell._ensure_gateway_session = shell.finalize_preloaded_skills = shell._install_tool_callbacks = lambda: None
        shell.conversation_history, shell.model, shell.session_id = [], "gpt-4o", "cli-session"
        shell.agent = shell._pending_title = shell._pending_one_turn_model_restore = shell._auto_load_skills_result = None
        shell._resumed = shell._single_query_mode = False
        built = SimpleNamespace(provider="openrouter", model="gpt-4o", session_id="cli-session", _print_fn=None)
        with patch("hermes_cli.gateway_agent.GatewayAgent", return_value=built), patch("hermes_cli.gateway_tools.bind"), \
                patch("hermes_cli.cli_agent_setup_mixin._current_runtime", return_value={"provider": "openrouter"}):
            assert shell._init_agent() is True
        assert built._route_resolution_entry["provider"] == "openai"

    def test_a_local_background_agent_gets_the_turn_check(self, hooks):
        hooks["route_custody"] = [_allow_resolve_only([])]
        shell, agent = _shell([]), _stub_agent()
        shell._route_entry_record = dict(RECORD)
        route_hooks._remember_session("cli-session", labels=frozenset({"acme:client"}))
        shell._bind_side_route(agent)
        assert route_hooks._session_labels(agent) == {"acme:client"}
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t")


# -- B2: side agents ------------------------------------------------------------------------------


def _side_authority(resident=None):
    authority = SimpleNamespace(runner=SimpleNamespace(_cached_agent_for=lambda _key: resident), db=None,
                                sessions={}, profile_id="p")
    return authority


def _side_policy():
    return SimpleNamespace(
        config=lambda _authority: {}, current_toolsets=lambda _config: [], max_turns=5, platform="tui",
        current_reasoning_config=lambda *_a: None, current_service_tier=lambda *_a: None, ignore_rules=False)


def _gateway_side_agent(record, resident=None, **overrides):
    """``_side_agent`` with its resolver stubbed: the resolver stashes *record*, as a fallback does."""
    from gateway import session_tui_side

    seen = {}

    def resolve(_runner, _policy, **_kw):
        seen["session"] = route_hooks._RESOLVING_SESSION.get()
        route_hooks.stash_resolution_entry(record)
        return "gpt-4o", {"provider": "openrouter", "api_key": "k", "base_url": "u"}

    def build(**kwargs):
        return _stub_agent(session_id=kwargs["session_id"])

    live = SimpleNamespace(route="key", source=None, superseded=False)
    with patch("gateway.run_turn_prepare._resolve_policy_agent_runtime", side_effect=resolve), \
            patch("run_agent.AIAgent", side_effect=build), patch("gateway.run._checkpoint_agent_kwargs", return_value={}), \
            patch("gateway.session_worker_construct.session_routing", return_value={}):
        return session_tui_side._side_agent(_side_authority(resident), _side_policy(), live, "bg_1", **overrides), seen


class TestGatewaySideAgent:
    def test_the_stashed_entry_is_bound_and_the_turn_is_refused(self, hooks):
        hooks["route_custody"] = [_allow_resolve_only([])]
        agent, _seen = _gateway_side_agent(RECORD)
        assert agent._route_resolution_entry["provider"] == "openai"
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t")

    def test_a_required_entry_with_no_gate_is_refused_at_its_turn(self, hooks):
        agent, _seen = _gateway_side_agent(RECORD)
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t")

    def test_the_resolve_phase_and_the_turn_read_the_parent_session(self, hooks):
        asked = []
        hooks["route_custody"] = [_allow_resolve_only(asked)]
        resident = _stub_agent(session_id="parent")
        resident._route_labels = ("parent", frozenset({"held:label"}))
        route_hooks._remember_session("parent", labels=frozenset({"acme:client"}),
                                      data_use=frozenset({TOOL_DATA}))
        agent, seen = _gateway_side_agent(RECORD, resident)
        assert seen["session"] == "parent"
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t")
        assert asked == [("turn", "openai", "", [TOOL_DATA], ["acme:client", "held:label"])]

    def test_no_gate_and_no_key_binds_nothing(self, hooks):
        agent, _seen = _gateway_side_agent(None)
        assert not hasattr(agent, "_route_resolution_entry") and not hasattr(agent, "_route_labels")


class TestGatewayBackground:
    def test_the_image_check_and_the_agent_read_the_parent_session(self, hooks):
        from tests.agent.test_route_hooks_vision_mutate import _background_runner, _run_background

        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        route_hooks._remember_session("parent", labels=frozenset({"acme:client"}))
        runner = _background_runner()
        runner.session_store.peek_session_id.return_value = "parent"
        _seen, built = _run_background(runner)
        assert asked and asked[0][4] == ["acme:client"]
        assert built._route_labels[1] == {"acme:client"}


class TestTuiSideAgents:
    def _server(self, monkeypatch, parent):
        from tui_gateway import server

        built = []

        class Agent:
            def __init__(self, **kwargs):
                self.provider, self.model, self.session_id = "openai", "gpt-4o", kwargs.get("session_id", "bg")
                self.platform, self._provider_fallback_active = "tui", False
                built.append(self)

            def run_conversation(self, **_kw):
                route_hooks.begin_turn(self, messages=[], turn_id="t")
                return {"final_response": "sent"}

            def close(self):
                pass

        monkeypatch.setattr("run_agent.AIAgent", Agent)
        monkeypatch.setattr(server, "_background_agent_kwargs", lambda *_a: {"session_id": "bg"})
        monkeypatch.setattr(server, "_ephemeral_preview_agent_kwargs", lambda *_a: {"session_id": "preview"})
        monkeypatch.setattr(server, "_preview_restart_callbacks", lambda *_a: {})
        bodies = []

        def spawn(_rid, _session, _task_id, _parent, _event, body, **_kw):
            bodies.append(body)
            return {}

        monkeypatch.setattr(server, "_spawn_side_agent", spawn)
        monkeypatch.setattr(server, "_side_agent_args", lambda *_a: ({"agent": parent}, "text", "s", "bg_1", None))
        monkeypatch.setattr(server, "_sess", lambda *_a: ({"agent": parent, "history": []}, None))
        return server, bodies, built

    def test_background_is_refused_on_the_parent_kept_fallback(self, hooks, monkeypatch):
        hooks["route_custody"] = [_skip_all([])]
        server, bodies, built = self._server(monkeypatch, _kept_fallback_parent(HOSTED))
        server._methods["prompt.background"]("rid", {"session_id": "s", "text": "x"})
        with pytest.raises(route_hooks.RouteCustodyRefused):
            bodies[0]()
        assert built[0]._route_resolution_entry["source"] == "side"

    def test_background_reads_the_parent_labels_and_data_use(self, hooks, monkeypatch):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        parent = _kept_fallback_parent(HOSTED)
        route_hooks._remember_session("parent", labels=frozenset({"acme:client"}), data_use=frozenset({TOOL_DATA}))
        server, bodies, _built = self._server(monkeypatch, parent)
        server._methods["prompt.background"]("rid", {"session_id": "s", "text": "x"})
        with pytest.raises(route_hooks.RouteCustodyRefused):
            bodies[0]()
        assert asked == [("turn", "openai", "", [TOOL_DATA], ["acme:client"])]

    def test_preview_restart_is_refused_on_the_parent_entry(self, hooks, monkeypatch):
        hooks["route_custody"] = [_allow_resolve_only([])]
        parent = _stub_agent(provider="openrouter", session_id="parent")
        route_hooks.bind_resolution_entry(parent, RECORD)
        server, bodies, _built = self._server(monkeypatch, parent)
        monkeypatch.setattr(server, "_validated_preview_cwd", lambda _cwd: "")
        monkeypatch.setattr(server, "_preview_restart_prompt", lambda *_a: "restart")
        monkeypatch.setattr(server, "_emit", lambda *_a: None)
        server._methods["preview.restart"]("rid", {"session_id": "s", "url": "http://localhost:1"})
        # The agent copied the parent's provider and model: the record's canonical provider matches it.
        with patch.object(route_hooks, "_provider_forms", side_effect=lambda name: frozenset({"openrouter", "openai"})):
            with pytest.raises(route_hooks.RouteCustodyRefused):
                bodies[0]()


class TestParentRecord:
    def test_a_kept_fallback_becomes_the_child_record(self):
        record = route_hooks.side_route_record(_kept_fallback_parent())
        assert record["route_custody"] == "required" and record["resolved_provider"] == "openai"

    def test_a_primary_has_no_record(self, hooks):
        assert route_hooks.side_route_record(_stub_agent()) is None

    def test_a_parent_that_left_the_entry_has_no_record(self, hooks):
        hooks["route_custody"] = [_skip_all([])]
        parent = _stub_agent(provider="zai", model="glm-4.7")
        parent._route_resolution_entry = dict(RECORD)
        assert route_hooks.side_route_record(parent) is None


# -- B2: side questions -----------------------------------------------------------------------------


class _Sends:
    """Stands in for both side-question paths and records what they would have sent."""

    def __init__(self, fork_result=None):
        self.fork, self.oneshot = [], []
        self.fork_result = fork_result

    def patches(self):
        def fork(parent_agent, question, history, on_fork=None):
            self.fork.append(question)
            if isinstance(self.fork_result, Exception):
                raise self.fork_result
            return self.fork_result or "fork answer"

        def oneshot(question, history, **_kw):
            self.oneshot.append(question)
            return "one-shot answer"

        return patch("agent.side_question._answer_via_fork", side_effect=fork), \
            patch("agent.side_question._answer_via_oneshot", side_effect=oneshot)


def _ask(sends, history=None, parent=None, runtime=None, route=None, slot=None):
    fork, oneshot = sends.patches()
    with fork, oneshot, patch("agent.side_question._side_question_task_config", return_value=slot or {}):
        return answer_side_question("why?", history if history is not None else _stamped_history(), parent_agent=parent,
                                    main_runtime=runtime or {"provider": "openrouter", "model": "gpt-4o", "session_id": "s1"},
                                    route=route)


class TestSideQuestionGate:
    def test_a_refused_entry_sends_nothing_on_the_one_shot(self, hooks):
        asked = []
        hooks["route_custody"] = [_allow_resolve_only(asked)]
        sends = _Sends()
        with pytest.raises(route_hooks.RouteCustodyRefused):
            _ask(sends, route={"record": dict(RECORD), "session_id": "s1", "platform": "telegram"})
        assert sends.oneshot == []
        assert asked == [("auxiliary", "openai", "side_question", [TOOL_DATA], [])]

    def test_a_required_entry_with_no_gate_sends_nothing(self, hooks):
        sends = _Sends()
        with pytest.raises(route_hooks.RouteCustodyRefused):
            _ask(sends, route={"record": dict(RECORD), "session_id": "s1"})
        assert sends.oneshot == []

    def test_a_refused_kept_fallback_sends_nothing_on_the_fork(self, hooks):
        hooks["route_custody"] = [_skip_all([])]
        sends = _Sends()
        with pytest.raises(route_hooks.RouteCustodyRefused):
            _ask(sends, parent=_kept_fallback_parent(HOSTED))
        assert sends.fork == [] and sends.oneshot == []

    def test_turn_route_labels_the_question(self, hooks):
        asked, routed = [], []

        def label(phase, reason, **_kw):
            routed.append((phase, reason))
            return {"labels": ["acme:client"]}

        hooks["turn_route"] = [label]
        hooks["route_custody"] = [_allow_resolve_only(asked)]
        with pytest.raises(route_hooks.RouteCustodyRefused):
            _ask(_Sends(), route={"record": dict(RECORD), "session_id": "s1"})
        assert routed == [("turn", "side_question")] and asked[0][4] == ["acme:client"]

    def test_a_failing_labeller_refuses(self, hooks):
        def broken(**_kw):
            raise RuntimeError("labeller down")

        hooks["turn_route"] = [broken]
        hooks["route_custody"] = [_allow_resolve_only([])]
        sends = _Sends()
        with patch("agent.route_hooks.announce_route", return_value=(frozenset(), False)), \
                pytest.raises(route_hooks.RouteCustodyRefused):
            _ask(sends, route={"record": dict(RECORD), "session_id": "s1"})
        assert sends.oneshot == []

    def test_a_fork_refused_at_its_turn_never_falls_to_the_one_shot(self, hooks):
        hooks["route_custody"] = [_allow_resolve_only([])]
        sends = _Sends(route_hooks.RouteCustodyRefused("refused at the fork's turn"))
        with pytest.raises(route_hooks.RouteCustodyRefused):
            _ask(sends, parent=_stub_agent(session_id="s1"))
        assert sends.fork and sends.oneshot == []

    def test_a_fork_that_failed_otherwise_keeps_the_one_shot(self, hooks):
        sends = _Sends(RuntimeError("provider down"))
        assert _ask(sends, parent=_stub_agent(session_id="s1")) == "one-shot answer"

    def test_an_admitted_route_answers(self, hooks):
        asked = []

        def allow(phase, **_kw):
            asked.append(phase)
            return {"action": "allow"}

        hooks["route_custody"] = [allow]
        assert _ask(_Sends(), route={"record": dict(RECORD), "session_id": "s1"}) == "one-shot answer"
        assert asked == ["auxiliary"]

    def test_no_gate_and_no_key_never_asks(self, hooks):
        with patch("agent.route_hooks._invoke", side_effect=AssertionError("no hook may run")):
            assert _ask(_Sends()) == "one-shot answer"

    def test_a_slot_off_the_main_route_is_not_the_entry(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        answer = _ask(_Sends(), route={"record": dict(RECORD), "session_id": "s1"},
                      slot={"provider": "zai", "model": "glm-4.7"})
        assert answer == "one-shot answer" and asked == []

    @pytest.mark.parametrize("slot", [{"provider": "zai"}, {"provider": "main", "model": "m"},
                                      {"provider": "openrouter", "model": "m"}, {"model": "m"}])
    def test_a_slot_the_fork_keeps_on_the_parent_route_is_checked(self, hooks, slot):
        hooks["route_custody"] = [_skip_all([])]
        with pytest.raises(route_hooks.RouteCustodyRefused):
            _ask(_Sends(), route={"record": dict(RECORD), "session_id": "s1"}, slot=slot)

    def test_an_unreadable_slot_is_checked(self, hooks):
        hooks["route_custody"] = [_skip_all([])]
        fork, oneshot = _Sends().patches()
        with fork, oneshot, patch("agent.side_question._side_question_task_config", side_effect=OSError("torn")), \
                pytest.raises(route_hooks.RouteCustodyRefused):
            answer_side_question("why?", [], main_runtime={"provider": "openrouter", "model": "gpt-4o"},
                                 route={"record": dict(RECORD), "session_id": "s1"})


class TestSideQuestionFork:
    def _parent(self, hooks):
        from tests.agent.test_route_hooks import _activate, _make_agent
        parent = _make_agent([dict(HOSTED)])
        parent.model, parent.provider = "anthropic/claude-sonnet-4", "openrouter"
        parent._primary_runtime = {**(parent._primary_runtime or {}), "provider": "openrouter",
                                   "model": parent.model}
        _activate(parent)
        return parent

    def _fork(self, parent, task_cfg=None):
        from agent.background_review import build_cache_parity_fork
        with patch("model_tools.get_tool_definitions", return_value=[]), \
                patch("model_tools.check_toolset_requirements", return_value={}), \
                patch("agent.process_bootstrap.OpenAI"):
            return build_cache_parity_fork(parent, task_cfg, max_iterations=2, write_origin="side_question")[0]

    def test_the_fork_carries_the_kept_fallback_to_its_turn_check(self, hooks):
        parent = self._parent(hooks)
        assert parent._provider_fallback_active
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        fork = self._fork(parent)
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(fork, messages=[], turn_id="t")
        assert [phase for phase, *_rest in asked] == ["turn"]

    def test_a_fork_on_a_primary_is_not_gated(self, hooks):
        from tests.agent.test_route_hooks import _make_agent
        parent = _make_agent(None)
        parent.provider, parent.model = "openrouter", "gpt-4o"
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        fork = self._fork(parent)
        route_hooks.begin_turn(fork, messages=[], turn_id="t")
        assert asked == []

    def test_a_refused_fork_result_raises(self, hooks):
        from agent import side_question
        fork = MagicMock()
        fork.run_conversation.return_value = {"failure_reason": "route_custody", "error": "not cleared"}
        with patch("agent.background_review.build_cache_parity_fork", return_value=(fork, {}, False)), \
                patch("agent.background_review._record_review_usage_to_parent"), \
                patch("agent.background_review._snapshot_review_usage"), \
                pytest.raises(route_hooks.RouteCustodyRefused):
            side_question._answer_via_fork(MagicMock(), "why?", [])


class TestGatewayBtw:
    def _runner(self, record):
        from tests.gateway.test_background_command import _make_runner
        runner = _make_runner()
        store = MagicMock()
        store.get_or_create_session = AsyncMock(return_value=SimpleNamespace(session_id="s1"))
        store.load_transcript = AsyncMock(return_value=_stamped_history())
        store._store = runner.session_store
        runner._async_session_store = store

        def resolve(**_kw):
            route_hooks.stash_resolution_entry(record)
            return "gpt-4o", {"api_key": "k", "provider": "openrouter", "base_url": "u", "api_mode": "chat_completions"}

        runner._resolve_session_agent_runtime = resolve
        runner._reply_anchor_for_event = MagicMock(return_value=None)
        runner._thread_metadata_for_source = runner._reply_metadata = MagicMock(return_value=None)
        runner._delivery_adapter_for = MagicMock(return_value=AsyncMock())
        return runner

    def _btw(self, runner):
        from tests.gateway.test_background_command import _make_event

        async def go():
            await runner._handle_btw_command(_make_event(text="/btw which file?"))
            await asyncio.gather(*list(runner._background_tasks))

        asyncio.run(go())
        return runner._delivery_adapter_for.return_value.send.call_args[0][1]

    def test_the_gate_is_asked_and_its_refusal_sends_nothing(self, hooks):
        asked = []
        hooks["route_custody"] = [_allow_resolve_only(asked)]
        sends = _Sends()
        fork, oneshot = sends.patches()
        with fork, oneshot:
            reply = self._btw(self._runner(dict(RECORD)))
        assert sends.oneshot == [] and "route custody refused" in reply
        assert asked == [("auxiliary", "openai", "side_question", [TOOL_DATA], [])]

    def test_a_resolved_primary_is_not_gated(self, hooks):
        asked = []
        hooks["route_custody"] = [_skip_all(asked)]
        sends = _Sends()
        fork, oneshot = sends.patches()
        with fork, oneshot:
            self._btw(self._runner(None))
        assert sends.oneshot and asked == []


class TestTuiBtw:
    def test_the_owner_path_hands_the_record_and_the_parent_session(self, hooks, monkeypatch):
        from gateway import session_tui_side
        from gateway.session_authority import SessionRef

        seen, resident = {}, _stub_agent(session_id="physical")
        resident._session_messages = [{"role": "user", "content": "hi"}]

        def answer(_text, _history, **kwargs):
            seen.update(kwargs)
            return "ok"

        def resolve(*_a):
            route_hooks.stash_resolution_entry(RECORD)
            return "m", {"provider": "openrouter"}

        events = []
        live = SimpleNamespace(task=None, route="key", source=None, superseded=False, event_stream=SimpleNamespace(
            execution={}, publish=lambda _sid, payload, event_type: events.append(payload)))
        authority = SimpleNamespace(sessions={"s": live}, pending_stops={}, adopted={}, agent=lambda _ref: resident)
        runner = SimpleNamespace(_running_agents={}, session_authority=authority, _persist_active_agents=lambda: None)
        authority.runner = runner
        monkeypatch.setattr(session_tui_side, "_admit", lambda *_: (live, SimpleNamespace(platform="tui"), "why?", "btw_x"))
        monkeypatch.setattr(session_tui_side, "_side_scope", lambda *_: contextlib.nullcontext())
        monkeypatch.setattr(session_tui_side, "_resolve_runtime", resolve)
        monkeypatch.setattr("agent.side_question.answer_side_question", answer)

        async def go():
            await session_tui_side.btw(SimpleNamespace(authority=authority), SessionRef("p", "s"), {})

        asyncio.run(go())
        import threading
        for thread in threading.enumerate():
            if thread.name == "side-agent-btw_x":
                thread.join(5)
        assert seen["route"]["record"]["provider"] == "openai" and seen["route"]["session_id"] == "physical"
