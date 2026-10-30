"""Route custody outside a turn and on the remaining resolution-time builders (agent/route_hooks.py).

The third review of TIN-7765 found these paths open at 57278f6b:

D: a compaction outside a turn (gateway ``/compress``, ``session.mutate compress``, gateway hygiene)
   runs in a fresh copied context, so the auxiliary gate saw no session and no data use.
A: Feishu comments, gateway ``/compress``, ``session.mutate compress`` and hygiene built on a
   resolution-time fallback and never bound its record; the gateway's resolve phase had no
   session id.
Race: the gateway handed the record over a runner-wide attribute that another session cleared.
Model rewrite: a model the agent rewrote after resolution (``nous/welcome``) skipped the check.
Tip adoption: adopting a live compaction tip did not carry the session's labels.

Each probe drives the production code with stub agents and patched resolvers. No provider is
called."""

import asyncio
import contextlib
import contextvars
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import route_hooks
from agent.message_metadata import stamp_data_use
from hermes_cli.auth import AuthError
from hermes_cli.plugins import PluginManager

HOSTED = {"provider": "openai", "model": "gpt-4o"}
REQUIRED = {**HOSTED, "route_custody": "required"}
# The resolver stores ``openai`` as ``openrouter``; the agent runs on the canonical id.
RECORD = route_hooks.resolution_record(REQUIRED, {"provider": "openrouter"})


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture(autouse=True)
def _fresh_route_context():
    tokens = [(var, var.set(None)) for var in (route_hooks._ROUTE_SCOPE, getattr(route_hooks, "_RESOLVED_ENTRY", None))
              if var is not None]
    yield
    for var, token in reversed(tokens):
        var.reset(token)


def _tool_message(requires):
    message = {"role": "tool", "content": "rows", "tool_call_id": "c1"}
    stamp_data_use(message, requires)
    return message


def _transcript():
    return [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            _tool_message(["acme:client-raw"]), {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"}]


def _make_agent(provider="openrouter", model="gpt-4o", session_id=None):
    from tests.agent.test_route_hooks import _make_agent as make
    agent = make(None)
    agent.provider, agent.model = provider, model
    if session_id is not None:
        agent.session_id = session_id
    return agent


def _hooks_md_gate(hooks, seen):
    """The labels-only example gate from the PR's earlier hooks.md, with ``openai`` as the hosted route."""
    hooks["turn_route"] = [lambda data_use, **_kw: {"labels": ["acme:client"]} if "acme:client-raw" in data_use else None]

    def custody(phase, candidate, labels, data_use, session_id, **_kw):
        seen.append((phase, candidate["provider"], labels, data_use, session_id))
        if phase == "resolve":
            return {"action": "allow"} if "acme:client" not in labels else {"action": "skip"}
        return {"action": "skip"} if "acme:client" in labels and candidate["provider"] == "openai" else {"action": "allow"}

    hooks["route_custody"] = [custody]


def _label_in_its_own_context(session_id):
    """A turn of *session_id* labels the session in its own copied context, as the gateway runs each turn."""
    turn_agent = _make_agent(session_id=session_id)
    contextvars.copy_context().run(route_hooks.begin_turn, turn_agent, messages=[_tool_message(["acme:client-raw"])],
                                   turn_id="t1")


class _SummaryRecorder:
    """Stands in for ``compress_context``: the summary call's auxiliary walk reaches a required hosted entry."""

    def __init__(self):
        self.calls, self.refused = 0, []

    def __call__(self, agent, messages, system_message, **_kw):
        self.calls += 1
        self.refused.append(route_hooks.auxiliary_refuses(REQUIRED, task="compression", failed_provider="anthropic"))
        return messages, system_message


@contextlib.contextmanager
def _compression_seam(recorder):
    # A positive idle timeout: the facade runs the summary on the compress-timeout pool.
    with patch("agent.conversation_compression.compress_context", recorder), \
         patch("agent.conversation_compression.resolve_context_compression_timeouts", return_value=(30.0, 60.0)):
        yield


# -- D: the auxiliary gate during a compaction outside a turn -------------------------------------


class TestOutOfTurnCompactionScope:
    def test_pool_thread_gate_sees_the_session_labels_and_transcript_data_use(self, hooks):
        seen, recorder = [], _SummaryRecorder()
        _hooks_md_gate(hooks, seen)
        _label_in_its_own_context("ooc-session")
        compacting = _make_agent(session_id="ooc-session")  # the gateway's temporary agent
        with _compression_seam(recorder):
            contextvars.copy_context().run(compacting._compress_context, _transcript(), "", task_id="ooc-session")
        assert recorder.refused == [True]
        assert seen[-1] == ("auxiliary", "openai", ["acme:client"], ["acme:client-raw"], "ooc-session")
        assert route_hooks._ROUTE_SCOPE.get() is None  # nothing leaks into the caller's context

    def test_transcript_data_use_reaches_a_data_use_gate_with_no_labels(self, hooks):
        recorder = _SummaryRecorder()
        hooks["route_custody"] = [lambda data_use, **_kw: {"action": "skip"} if "acme:client-raw" in data_use
                                  else {"action": "allow"}]
        compacting = _make_agent(session_id="never-labelled")
        with _compression_seam(recorder):
            contextvars.copy_context().run(compacting._compress_context, _transcript(), "")
        assert recorder.refused == [True]

    def test_in_turn_compaction_keeps_the_turn_scope(self, hooks):
        seen, recorder = [], _SummaryRecorder()
        _hooks_md_gate(hooks, seen)
        agent = _make_agent(session_id="in-turn")
        route_hooks.begin_turn(agent, messages=[_tool_message(["acme:client-raw"])], turn_id="t1")
        with _compression_seam(recorder):
            agent._compress_context(_transcript(), "")
        assert recorder.refused == [True]
        assert [s[0] for s in seen] == ["auxiliary"]  # no out-of-turn route check inside a turn
        assert route_hooks._scope_agent() is agent


# -- A: out-of-turn compaction on a resolution-time entry -------------------------------------------


def _resolve_allows_compaction_refuses(asked):
    def custody(phase, candidate, **_kw):
        asked.append((phase, candidate["provider"]))
        return {"action": "allow"} if phase == "resolve" else {"action": "skip"}
    return custody


class TestOutOfTurnRouteCheck:
    def test_compaction_on_a_refused_resolution_entry_sends_nothing(self, hooks):
        asked, recorder = [], _SummaryRecorder()
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        agent = _make_agent()
        route_hooks.bind_resolution_entry(agent, RECORD)
        with _compression_seam(recorder), pytest.raises(route_hooks.RouteCustodyRefused):
            contextvars.copy_context().run(agent._compress_context, _transcript(), "")
        assert recorder.calls == 0
        assert asked == [("compaction", "openai")]

    def test_the_gate_sees_the_transcript_at_the_compaction_phase(self, hooks):
        seen = []

        def custody(phase, messages, data_use, **_kw):
            seen.append((phase, len(messages), data_use))
            return {"action": "allow"}

        hooks["route_custody"] = [custody]
        recorder = _SummaryRecorder()
        agent = _make_agent()
        route_hooks.bind_resolution_entry(agent, RECORD)
        with _compression_seam(recorder):
            contextvars.copy_context().run(agent._compress_context, _transcript(), "")
        assert seen[0] == ("compaction", 5, ["acme:client-raw"])
        assert recorder.calls == 1

    def test_with_no_record_the_compaction_runs_and_a_required_aux_entry_still_needs_its_gate(self, hooks):
        recorder = _SummaryRecorder()
        agent = _make_agent()
        with _compression_seam(recorder):
            contextvars.copy_context().run(agent._compress_context, _transcript(), "")
        assert recorder.calls == 1 and recorder.refused == [True]


# -- A: each surface that builds on a resolution-time fallback -------------------------------------


def _fallback_runtime():
    return {"api_key": "k", "provider": "openrouter", "model": "gpt-4o", route_hooks.ROUTE_ENTRY_KEY: dict(RECORD)}


class TestGatewayCompressSurface:
    def test_manual_compress_on_a_refused_entry_fails_without_a_summary(self, hooks):
        from tests.gateway.test_compress_command import _make_event, _make_history, _make_runner

        asked, recorder = [], _SummaryRecorder()
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        agent = _make_agent(session_id="sess-1")
        agent.context_compressor.has_content_to_compress = lambda *_a, **_kw: True
        runner = _make_runner(_make_history())
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=_fallback_runtime()), \
             patch("gateway.run._resolve_gateway_model", return_value="gpt-5.6-sol"), \
             patch("run_agent.AIAgent", return_value=agent), _compression_seam(recorder):
            reply = asyncio.run(runner._handle_compress_command(_make_event()))
        assert recorder.calls == 0
        assert ("compaction", "openai") in asked
        assert "route custody refused" in reply


class TestHygieneSurface:
    def test_detached_hygiene_on_a_refused_entry_sends_nothing(self, hooks):
        from gateway.run import GatewayRunner

        asked, recorder = [], _SummaryRecorder()
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        agent = _make_agent(session_id="sess-1")

        async def _build_agent(*_args):
            return agent, None

        async def _wait_for_summary(attempt, *_args):
            return await attempt.future

        async def _cleanup(*_args, **_kwargs):
            return None

        gateway = SimpleNamespace(
            _hmwa_hygiene_build_agent=_build_agent, _hmwa_hygiene_wait_for_summary=_wait_for_summary,
            _hmwa_hygiene_on_unwind=lambda *_a: None, _track_deferred_agent_worker=lambda *_a: None,
            _evict_cached_agent=lambda _key: None, _cleanup_agent_resources_off_loop=_cleanup)
        attempt = GatewayRunner._HygieneAttempt(agent=None, meta=None)
        with _compression_seam(recorder), pytest.raises(route_hooks.RouteCustodyRefused):
            asyncio.run(GatewayRunner._hmwa_hygiene_detached_attempt(
                gateway, attempt, SimpleNamespace(total_ceiling_seconds=60.0), SimpleNamespace(approx_tokens=1000),
                _transcript(), _transcript(), "gpt-4o", {}, None, SimpleNamespace(session_id="sess-1"), "tg:1", "qk", 0,
                route_entry=dict(RECORD)))
        assert recorder.calls == 0
        assert asked == [("compaction", "openai")]


class _FakeCompressor:
    instances = []

    def __init__(self, *_args, **kwargs):
        self.provider = kwargs.get("provider")
        self.refused = []
        _FakeCompressor.instances.append(self)

    def compress(self, messages, **_kw):
        self.refused.append(route_hooks.auxiliary_refuses(REQUIRED, task="compression", failed_provider=self.provider))
        return messages


def _prepare_compress(record, history, session_id):
    from gateway.session_mutation_compress import prepare_compress

    def resolve(**_kw):  # the gateway resolver's contract: stash the record in this context
        route_hooks.stash_resolution_entry(record)
        return "gpt-4o", {"provider": "openrouter", "api_key": "k"}

    authority = MagicMock()
    authority.runner._resolve_session_agent_runtime.side_effect = resolve
    authority.runner._cached_agent_for = lambda _key: None
    authority.db.get_messages_as_conversation.return_value = history
    authority.db.get_session.return_value = None
    policy = SimpleNamespace(config=lambda _authority: {}, safe_mode=True)
    prepared = {"snapshot": {"target": session_id, "receipt": {"policy": {}}}}
    _FakeCompressor.instances.clear()
    with patch("gateway.session_policy.restore_policy", return_value=policy), \
         patch("gateway.session_policy.policy_scope", lambda *_a, **_kw: contextlib.nullcontext()), \
         patch("agent.context_compressor.ContextCompressor", _FakeCompressor):
        return asyncio.run(prepare_compress(authority, SimpleNamespace(source=None, route="key"), {}, prepared))


class TestSessionMutateSurface:
    def test_refused_entry_refuses_the_mutation_before_any_summary(self, hooks):
        from hermes_state_runtime import RuntimeStoreError

        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        with pytest.raises(RuntimeStoreError) as refused:
            _prepare_compress(dict(RECORD), _transcript(), "mutate-session")
        assert refused.value.reason == "route_custody_refused"
        assert _FakeCompressor.instances == []
        assert asked == [("compaction", "openai")]

    def test_summary_walk_sees_the_session_labels_and_transcript_data_use(self, hooks):
        seen = []
        _hooks_md_gate(hooks, seen)
        _label_in_its_own_context("mutate-labelled")
        result = _prepare_compress(None, _transcript(), "mutate-labelled")
        assert result["status"] == "noop"
        assert _FakeCompressor.instances[0].refused == [True]
        assert seen[-1] == ("auxiliary", "openai", ["acme:client"], ["acme:client-raw"], "mutate-labelled")


class TestFeishuSurface:
    def test_comment_agent_carries_the_record_into_its_turn(self, hooks):
        from plugins.platforms.feishu import feishu_comment as fc

        asked, built = [], []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]

        class _Agent:
            def __init__(self, **kwargs):
                self.provider, self.model, self.session_id = kwargs["provider"], kwargs["model"], None
                built.append(self)

            def run_conversation(self, prompt, conversation_history=None):
                route_hooks.begin_turn(self, messages=[{"role": "user", "content": prompt}], turn_id="t1")
                return {"final_response": "sent"}

            def close(self):
                pass

        runtime = _fallback_runtime()
        model = runtime.pop("model")
        with patch.object(fc, "_resolve_model_and_runtime", return_value=(model, runtime)), \
             patch("run_agent.AIAgent", _Agent):
            assert fc._run_comment_agent("client rows", object()) == ""  # the turn was refused
        assert built[0]._route_resolution_entry["provider"] == "openai"
        assert asked == [("turn", "openai")]


# -- The gateway's record and session id travel with the resolving context -----------------------


def _gateway_resolution(runner, config):
    def resolve(**kw):
        if kw.get("requested") is None:
            raise AuthError("expired")
        return {"provider": "openrouter", "api_key": "k", "base_url": "u", "api_mode": "chat_completions"}

    return contextlib.ExitStack(), [
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve),
        patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "openai-codex", "default": "m"}),
        patch("gateway.run._load_gateway_config", return_value=config),
        patch("gateway.run._resolve_gateway_model", return_value="m"),
        patch("gateway.run._get_channel_override", return_value=None)]


class TestGatewayRecordHandOff:
    def _resolve(self, runner, config):
        stack, patches = _gateway_resolution(runner, config)
        with stack:
            for p in patches:
                stack.enter_context(p)
            runner._resolve_session_agent_runtime(session_key="test-session-key")
        return route_hooks.take_resolution_entry()

    def test_a_concurrent_resolution_never_clears_another_sessions_record(self, hooks):
        from tests.gateway.test_pre_agent_fallback_notice import _runner_with_real_runtime_resolution

        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        runner = _runner_with_real_runtime_resolution()
        fallback = {"fallback_providers": [REQUIRED]}
        session_x, session_y = contextvars.copy_context(), contextvars.copy_context()
        stack, patches = _gateway_resolution(runner, fallback)
        with stack:
            for p in patches:
                stack.enter_context(p)
            session_x.run(runner._resolve_session_agent_runtime, session_key="x")  # X falls back
            with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       return_value={"provider": "anthropic", "api_key": "k"}):
                session_y.run(runner._resolve_session_agent_runtime, session_key="y")  # Y resolves its primary
        assert session_x.run(route_hooks.take_resolution_entry)["provider"] == "openai"
        assert session_y.run(route_hooks.take_resolution_entry) is None
        assert not isinstance(getattr(runner, "_pre_agent_fallback_entry", None), dict)

    def test_the_resolve_phase_reads_the_sessions_earlier_labels(self, hooks):
        from tests.gateway.test_pre_agent_fallback_notice import _runner_with_real_runtime_resolution

        seen = []
        _hooks_md_gate(hooks, seen)
        route_hooks._remember_session_labels("resolve-session", frozenset({"acme:client"}))
        runner = _runner_with_real_runtime_resolution()
        runner.session_store.peek_session_id.return_value = "resolve-session"
        with pytest.raises(RuntimeError):  # the only entry is refused for a labelled session
            self._resolve(runner, {"fallback_providers": [HOSTED]})
        assert seen == [("resolve", "openai", ["acme:client"], [], "resolve-session")]


# -- The route the agent was built on ---------------------------------------------------------------


class TestBoundRoute:
    def test_a_model_pinned_after_construction_is_still_checked(self, hooks):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        entry = {"provider": "nous", "model": "Hermes-4-405B", "route_custody": "required"}
        agent = _make_agent(provider="nous", model="nous/welcome")  # pin_model_for_route ran in AIAgent()
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(entry, {"provider": "nous"}))
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "rows"}], turn_id="t1")
        assert asked == [("turn", "nous")]

    def test_a_deliberate_switch_leaves_the_entry_with_one_warning(self, hooks, caplog):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        agent = _make_agent()
        route_hooks.bind_resolution_entry(agent, RECORD)
        agent.provider, agent.model = "anthropic", "claude-opus-4"  # /model
        with caplog.at_level(logging.WARNING, logger="agent.route_hooks"):
            for turn in ("t1", "t2"):
                route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "q"}], turn_id=turn)
        assert asked == []
        assert sum("left the resolution-time entry" in r.getMessage() for r in caplog.records) == 1

    def test_an_agent_built_on_another_provider_is_not_the_entry(self, hooks):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        agent = _make_agent(provider="anthropic", model="claude-opus-4")  # a channel's own provider won
        route_hooks.bind_resolution_entry(agent, RECORD)
        assert "bound_route" not in agent._route_resolution_entry
        route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "q"}], turn_id="t1")
        assert asked == []

    def test_the_init_record_settles_once_on_the_built_route(self):
        agent = _make_agent(model="gpt-4o")
        route_hooks.bind_resolution_entry(agent, RECORD, source="init")
        agent.model = "openai/gpt-4o"  # construction rewrote the model after the walker chose the entry
        route_hooks.settle_init_resolution_route(agent)
        agent.model = "something-else"
        route_hooks.settle_init_resolution_route(agent)
        assert agent._route_resolution_entry["bound_route"] == {"provider": "openrouter", "model": "openai/gpt-4o"}


# -- Tip adoption carries the session's labels ------------------------------------------------------


class _TipDB:
    def get_compression_tip(self, session_id):
        return "tip-child"

    def get_session(self, session_id):
        return {"id": session_id, "ended_at": None, "system_prompt": None}

    def get_messages_as_conversation(self, session_id):
        return [{"role": "user", "content": "kept"}]


class TestTipAdoption:
    def test_adopting_the_tip_carries_labels_the_registry_evicted(self):
        from agent.conversation_compression import _adopt_live_compression_child

        agent = _make_agent(session_id="tip-parent")
        # Another agent instance labelled the parent, and the registry has since evicted it: only this
        # agent's own held copy remains.
        agent._route_labels = ("tip-parent", frozenset({"acme:client"}))
        with route_hooks._SESSIONS_LOCK:
            route_hooks._SESSIONS.pop("tip-parent", None)
        assert _adopt_live_compression_child(agent, _TipDB(), "tip-parent") is not None
        assert agent.session_id == "tip-child"
        assert route_hooks.session_labels("tip-child") == frozenset({"acme:client"})
