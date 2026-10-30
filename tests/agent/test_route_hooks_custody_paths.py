"""Route custody on the paths the second review of TIN-7765 found open (agent/route_hooks.py).

A: an agent built on a resolution-time fallback whose provider the resolver normalized (``openai``
   to ``openrouter``, ``claude`` to ``anthropic``) is re-checked at every turn start, on the
   gateway, TUI and cron surfaces, with no help from the agent's own fallback chain.
B: a custody restore at turn start publishes the restored primary for auxiliary ``auto`` calls.
C: a compaction that rotates the session id keeps the session's labels and data use.
D: an auxiliary call on the compression pool sees the turn's session, labels and data use.

Each surface probe drives the production builder with a stub agent and a patched resolver. No
provider is called."""

import concurrent.futures
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import route_hooks
from agent.error_classifier import FailoverReason
from agent.message_metadata import stamp_data_use
from hermes_cli.auth import AuthError
from hermes_cli.plugins import PluginManager

HOSTED = {"provider": "openai", "model": "gpt-4o", "api_key": "fb-secret"}
LOCAL = {"provider": "zai", "model": "glm-4.7", "api_key": "local-key"}
REPLY = "Holding: no cleared route."


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture(autouse=True)
def _fresh_route_scope():
    token = route_hooks._ROUTE_SCOPE.set(None)
    yield
    route_hooks._ROUTE_SCOPE.reset(token)


def _resolve_allows_turn_refuses(asked):
    """A gate that admits at resolve time (labels only) and refuses once the turn is visible."""
    def custody(phase, candidate, **_kw):
        asked.append((phase, candidate["provider"]))
        return {"action": "allow"} if phase == "resolve" else {"action": "skip"}
    return custody


def _tool_message(requires):
    message = {"role": "tool", "content": "rows", "tool_call_id": "c1"}
    stamp_data_use(message, requires)
    return message


def _make_agent(fallback_model):
    from tests.agent.test_route_hooks import _make_agent as make
    return make(fallback_model)


def _activate(agent):
    from tests.agent.test_route_hooks import _activate as activate
    return activate(agent)


# ── A: resolution-time entries, recorded by each walker ──────────────────────────────────────


class TestGatewaySurface:
    def test_aliased_entry_is_rechecked_at_turn_start(self, hooks):
        from gateway.run_turn_runner import TurnRunner
        from gateway.session import Platform, SessionSource
        from gateway.turn_context import TurnContext
        from tests.gateway.test_pre_agent_fallback_notice import _RecordingAgent, _runner_with_real_runtime_resolution

        class _Agent(_RecordingAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.provider = kwargs.get("provider")

        asked = []
        hooks["route_custody"] = [_resolve_allows_turn_refuses(asked)]

        def resolve(**kw):
            if kw.get("requested") is None:
                raise AuthError("expired")
            # The resolver stores the canonical id: ``openai`` resolves to ``openrouter``.
            return {"provider": "openrouter", "api_key": "k", "base_url": "u", "api_mode": "chat_completions"}

        runner = _runner_with_real_runtime_resolution()
        ctx = TurnContext(
            source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
            message="hi", history=[], session_id="sid", session_key="test-session-key", user_config={},
            AIAgent=_Agent, resolve_display_setting=lambda *_a: False, _run_still_current=lambda: True,
            _hooks_ref=SimpleNamespace(loaded_hooks=False))
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
             patch("hermes_cli.runtime_provider._get_model_config",
                   return_value={"provider": "openai-codex", "default": "gpt-5.6-sol"}), \
             patch("gateway.run._load_gateway_config",
                   return_value={"fallback_providers": [{"provider": "openai", "model": "gpt-4o"}]}), \
             patch("gateway.run._resolve_gateway_model", return_value="gpt-5.6-sol"), \
             patch("gateway.run._get_channel_override", return_value=None):
            TurnRunner(runner, ctx).run_sync()
        agent = ctx.agent_holder[0]
        assert (agent.provider, agent.model) == ("openrouter", "gpt-4o")
        assert route_hooks.ROUTE_ENTRY_KEY not in _Agent.built_kwargs  # private: never an AIAgent kwarg
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "client rows"}], turn_id="t1")
        assert asked == [("resolve", "openai"), ("turn", "openai")]

    def test_a_turn_whose_primary_resolved_clears_the_record(self, hooks):
        agent = _make_agent(None)
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(HOSTED, {"provider": "openrouter"}))
        route_hooks.bind_resolution_entry(agent, None)
        assert agent._route_resolution_entry is None
        # An init-time record belongs to the agent's own walker, not to the gateway's resolution.
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(HOSTED), source="init")
        route_hooks.bind_resolution_entry(agent, None)
        assert agent._route_resolution_entry["provider"] == "openai"


class _TuiAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.provider, self.model, self.session_id = kwargs["provider"], kwargs["model"], kwargs.get("session_id")


class TestTuiSurface:
    def test_aliased_entry_is_rechecked_at_turn_start(self, hooks, monkeypatch, tmp_path):
        from tui_gateway import server

        asked = []
        hooks["route_custody"] = [_resolve_allows_turn_refuses(asked)]
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
        (tmp_path / "config.yaml").write_text("model:\n  default: gpt-5.6-sol\n  provider: openai-codex\n")
        monkeypatch.setattr(server, "_hermes_home", tmp_path)
        monkeypatch.setattr(server, "_get_db", lambda: None)
        monkeypatch.setattr(server, "_load_fallback_model",
                            lambda: [{"provider": "claude", "model": "claude-sonnet-5", "api_key": "fb"}])

        def resolve(**kw):
            if kw.get("requested") != "claude":
                raise AuthError("expired")
            return {"api_key": "k", "base_url": "https://example.invalid/v1", "provider": "anthropic",
                    "api_mode": "anthropic_messages"}

        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
             patch("run_agent.AIAgent", _TuiAgent):
            agent = server._make_agent("sid", "key", context_cwd_is_launch_artifact=False)
        assert agent.provider == "anthropic"
        assert route_hooks.ROUTE_ENTRY_KEY not in agent.kwargs
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "client rows"}], turn_id="t1")
        assert asked == [("resolve", "claude"), ("turn", "claude")]


def _custody_turn(agent):
    """What run_conversation does with the turn-start custody check (agent/conversation_loop.py)."""
    try:
        route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "client rows"}], turn_id="t1")
    except route_hooks.RouteCustodyRefused as exc:
        return route_hooks.route_custody_refused_result(agent, exc, [])
    return {"final_response": "ok", "messages": []}


class TestCronSurface:
    def test_aliased_entry_is_rechecked_and_the_gate_reply_is_delivered(self, hooks, tmp_path):
        from tests.cron.test_cron_pre_agent_fallback_notice import _JOB
        from tests.cron.test_scheduler import _run_owned_job

        asked = []
        hooks["route_custody"] = [_resolve_allows_turn_refuses(asked)]
        hooks["on_routes_exhausted"] = [lambda **_kw: {"reply": REPLY}]
        (tmp_path / "config.yaml").write_text(
            "model:\n  default: gpt-5.6-sol\n  provider: openai-codex\n"
            "fallback_providers:\n  - provider: claude\n    model: claude-sonnet-5\n    api_key: fb-key\n")

        def resolve(**kw):
            if kw.get("requested") != "claude":
                raise AuthError("expired")
            return {"api_key": "k", "base_url": "https://example.invalid/v1", "provider": "anthropic",
                    "api_mode": "anthropic_messages"}

        fake_db = MagicMock()
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
             patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state_registry.acquire", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
             patch("run_agent.AIAgent") as agent_cls:
            agent = agent_cls.return_value
            agent.provider, agent.model, agent.session_id, agent.platform = "anthropic", "claude-sonnet-5", "cron-s", "cron"
            agent._provider_fallback_active, agent._fallback_chain, agent._fallback_index = False, [], 0
            agent._session_messages = agent._route_labels = agent._route_data_use = None
            agent._primary_runtime = {}
            agent.run_conversation.side_effect = lambda *_a, **_kw: _custody_turn(agent)
            success, _output, final, error = _run_owned_job(dict(_JOB), tmp_path, fake_db)
        assert route_hooks.ROUTE_ENTRY_KEY not in agent_cls.call_args.kwargs
        assert asked == [("resolve", "claude"), ("turn", "claude")]
        assert success is False and final == ""
        assert REPLY in error  # the gate's F6 reply, not the default copy


class TestRecordNotChain:
    def test_safe_mode_or_api_locked_agent_is_still_checked(self, hooks):
        """Safe mode and an API runtime lock leave the agent with no fallback chain; the record
        alone drives the turn-start check."""
        required = {**HOSTED, "route_custody": "required"}
        agent = _make_agent(None)
        agent.provider, agent.model = "openrouter", "gpt-4o"
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(required, {"provider": "openrouter"}))
        assert not agent._fallback_chain
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t1")

    def test_a_primary_equal_to_a_chain_entry_is_not_treated_as_a_fallback(self, hooks):
        """Without a record nothing is inferred: a primary that equals a required entry keeps working."""
        required = {**HOSTED, "route_custody": "required"}
        agent = _make_agent([required, LOCAL])
        agent.provider, agent.model = "openai", "gpt-4o"
        route_hooks.begin_turn(agent, messages=[], turn_id="t1")  # no RouteCustodyRefused

    def test_a_record_stops_applying_once_the_agent_moves_off_it(self, hooks):
        required = {**HOSTED, "route_custody": "required"}
        agent = _make_agent(None)
        agent.provider, agent.model = "openrouter", "gpt-4o"
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(required, {"provider": "openrouter"}))
        agent.provider, agent.model = "zai", "glm-4.7"  # a /model switch
        route_hooks.begin_turn(agent, messages=[], turn_id="t1")


# ── B: the restored primary is published again ───────────────────────────────────────────────


class TestRestorePublishes:
    def test_auxiliary_main_route_follows_a_custody_restore(self, hooks):
        from tests.agent.test_route_hooks import TestTurnStartOrder, _KeptFallbackAgent

        hooks["route_custody"] = [lambda candidate, **_kw: {"action": "skip"} if candidate["provider"] == "openai" else None]
        agent = _KeptFallbackAgent(restorable=True)
        published = []
        with patch("agent.auxiliary_client.set_runtime_main", lambda provider, model, **_kw: published.append((provider, model))):
            TestTurnStartOrder()._turn(agent)
        assert (agent.provider, agent.model) == ("anthropic", "opus")
        assert published[-1] == ("anthropic", "opus")  # auto auxiliary calls leave the refused route


# ── C: compaction rotation keeps labels and data use ─────────────────────────────────────────


def _hooks_md_gate(hooks, seen=None):
    """The example gate in website/docs/user-guide/features/hooks.md, with ``openai`` as the hosted route."""
    hooks["turn_route"] = [lambda data_use, **_kw: {"labels": ["acme:client"]} if "acme:client-raw" in data_use else None]

    def custody(candidate, labels, **kw):
        if seen is not None:
            seen.append((kw.get("phase"), candidate["provider"], labels, kw.get("data_use"), kw.get("session_id")))
        return ({"action": "skip", "reason": "client data"} if "acme:client" in labels and candidate["provider"] == "openai"
                else {"action": "allow"})

    hooks["route_custody"] = [custody]


class TestCompactionRotation:
    def test_labels_and_data_use_follow_the_child_session(self, hooks):
        _hooks_md_gate(hooks)
        agent = _make_agent([{**HOSTED, "route_custody": "required"}, LOCAL])
        agent.session_id = "parent"
        route_hooks.begin_turn(agent, messages=[_tool_message(["acme:client-raw"])], turn_id="t1")
        # The rotation in conversation_compression: new id, then the carry.
        agent.session_id = "child"
        from agent.conversation_compression import _carry_session_state_to_child
        with patch("agent.compression_automation.carry_compression_automation"):
            _carry_session_state_to_child(agent, "parent", None)
        assert route_hooks._session_labels(agent) == {"acme:client"}
        assert route_hooks._session_data_use(agent) == {"acme:client-raw"}
        # The summary rows carry no data_use stamp; the next turn still refuses the hosted entry.
        route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "[summary]"}], turn_id="t2")
        assert _activate(agent) is True
        assert (agent.provider, agent.model) == ("zai", "glm-4.7")


# ── D: the compression pool sees the turn ────────────────────────────────────────────────────


class TestCompressionPool:
    def test_pool_thread_gate_sees_the_turn_session_labels_and_data_use(self, hooks):
        from hermes_logging import set_session_context
        from tools.thread_context import propagate_context_to_thread

        seen = []
        _hooks_md_gate(hooks, seen)
        agent = _make_agent([HOSTED, LOCAL])
        agent.session_id = "client-session"
        route_hooks.begin_turn(agent, messages=[_tool_message(["acme:client-raw"])], turn_id="t1")
        required = {**HOSTED, "route_custody": "required"}

        def summary_call():
            set_session_context("last-child-this-thread-served")  # a stale thread binding is ignored
            return route_hooks.auxiliary_refuses(required, task="compression", failed_provider="anthropic")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            refused = pool.submit(propagate_context_to_thread(summary_call)).result(timeout=10)
        assert refused is True
        assert seen[-1] == ("auxiliary", "openai", ["acme:client"], ["acme:client-raw"], "client-session")

    def test_request_messages_carry_their_data_use_without_a_turn(self, hooks):
        seen = []
        hooks["route_custody"] = [lambda data_use, **_kw: seen.append(data_use) or {"action": "allow"}]
        with route_hooks.auxiliary_request([_tool_message(["acme:client-raw"])]):
            assert route_hooks.auxiliary_refuses(HOSTED, task="vision", failed_provider="anthropic") is False
        assert seen == [["acme:client-raw"]]


# ── Earlier findings 5 and 6: the refusal on each surface, and the billing nit ───────────────


def _refused_result(hooks):
    hooks["on_routes_exhausted"] = [lambda **_kw: {"reply": REPLY}]
    agent = _make_agent([HOSTED])
    return route_hooks.route_custody_refused_result(agent, route_hooks.RouteCustodyRefused("refused"), [])


class TestRefusalSurfaces:
    def test_gateway_delivers_the_gate_reply(self, hooks):
        from gateway.run import _normalize_empty_agent_response

        result = _refused_result(hooks)
        assert _normalize_empty_agent_response(result, result["final_response"]) == REPLY

    def test_tui_renders_the_gate_reply_as_an_error_turn(self, hooks):
        from tui_gateway.prompt_turn import _turn_outcome

        raw, status, _reasoning = _turn_outcome(_refused_result(hooks))
        assert (raw, status) == (REPLY, "error")

    def test_cron_fails_the_run_with_the_gate_reply(self, hooks):
        from cron.scheduler import _final_response_from_result

        with pytest.raises(RuntimeError, match=REPLY):
            _final_response_from_result(_refused_result(hooks), "j1", "job", None)

    def test_cron_without_a_reply_keeps_the_error_copy(self, hooks):
        from cron.scheduler import _final_response_from_result

        agent = _make_agent([HOSTED])
        result = route_hooks.route_custody_refused_result(agent, route_hooks.RouteCustodyRefused("refused"), [])
        with pytest.raises(RuntimeError, match="not sent"):
            _final_response_from_result(result, "j1", "job", None)


class TestBillingCopy:
    def test_a_402_keeps_its_billing_guidance(self, hooks):
        hooks["on_routes_exhausted"] = [lambda **_kw: {"reply": "busy"}]
        agent = _make_agent([HOSTED])
        assert _activate(agent) is True
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False
        classified = SimpleNamespace(reason=FailoverReason.billing, retryable=False,
                                     error_context={"reset_at": time.time() + 60})
        result = route_hooks.routes_exhausted(agent, {"final_response": "Add credits"}, classified=classified,
                                              provider="openai", model="gpt-4o")
        assert result["final_response"] == "Add credits" and result["routes_exhausted"] is True
