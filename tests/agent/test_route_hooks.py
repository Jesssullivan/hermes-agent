"""Route hooks (agent/route_hooks.py): turn_route labels a turn, route_custody skips a fallback
candidate per turn and fails closed, on_routes_exhausted replaces the exhausted copy. Each hook
does nothing until a plugin registers it or a fallback entry sets ``route_custody: required``.

The plugin side is a fake: callbacks registered on a fresh ``PluginManager``. No provider is
called; fallback clients come from a patched ``resolve_provider_client``."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import route_hooks
from agent.error_classifier import FailoverReason
from agent.message_metadata import stamp_data_use
from hermes_cli.auth import AuthError
from hermes_cli.plugins import PluginManager, SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS
from hermes_cli.plugins_dispatch import (
    _HOOK_TIMEOUT_BOUNDED_HOOKS, _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS,
)
from hermes_cli.runtime_provider import resolve_runtime_with_fallback
from run_agent import AIAgent

HOSTED = {"provider": "openai", "model": "gpt-4o", "api_key": "fb-secret"}
LOCAL = {"provider": "zai", "model": "glm-4.7", "api_key": "local-key"}


@pytest.fixture
def hooks(monkeypatch):
    """A fresh plugin manager standing in for the loaded plugins; returns its hook table."""
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


def _make_agent(fallback_model):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True, fallback_model=fallback_model,
        )
    agent.client = MagicMock()
    return agent


def _activate(agent):
    def client_for(provider=None, model=None, **_kw):
        client = MagicMock()
        client.base_url, client.api_key = f"https://{provider}.example/v1", "k"
        return client, model
    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=client_for):
        return agent._try_activate_fallback(reason=FailoverReason.rate_limit)


def _tool_message(requires):
    message = {"role": "tool", "content": "rows", "tool_call_id": "c1"}
    stamp_data_use(message, requires)
    return message


# ── Capability table and registration (F1 to F6 advertisement) ─────────────────────────


class TestCapabilities:
    def test_each_route_hook_is_advertised_and_registered(self):
        caps = route_hooks.HERMES_ROUTE_CAPABILITIES
        assert set(caps) == {"F1", "F2", "F3", "F6"}
        assert (caps["F2"], caps["F3"], caps["F6"]) == ("turn_route", "route_custody", "on_routes_exhausted")
        for hook in route_hooks.ROUTE_HOOKS:
            assert hook in VALID_HOOKS
            assert hook in SHELL_UNSUPPORTED_HOOKS  # no shell channel for labels, allow or reply
        assert "route_custody" in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        assert {"turn_route", "on_routes_exhausted"} <= _HOOK_TIMEOUT_BOUNDED_HOOKS
        with pytest.raises(TypeError):
            caps["F4"] = "x"  # read-only

    def test_f1_keys_exist_and_default_to_the_old_behaviour(self):
        """F1 is hermes#1: two keys that keep auxiliary and compression traffic off the main model.
        Both default true (unchanged behaviour); a home that needs F1 must render them false."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from agent.auxiliary_task_config import main_model_fallback_refused

        assert DEFAULT_CONFIG["auxiliary"]["main_model_fallback"] is True
        assert DEFAULT_CONFIG["compression"]["main_model_fallback"] is True
        with patch("agent.auxiliary_task_config._auxiliary_section", return_value={"main_model_fallback": False}):
            assert main_model_fallback_refused("compression", "rate limit", "custom") is True
        with patch("agent.auxiliary_task_config._auxiliary_section", return_value={}):
            assert main_model_fallback_refused("compression", "rate limit", "custom") is False


# ── Off by default ─────────────────────────────────────────────────────────────────────


class TestOffByDefault:
    def test_plain_chain_is_unchanged_without_a_plugin(self, hooks):
        agent = _make_agent([HOSTED, LOCAL])
        with patch.object(route_hooks, "_invoke") as invoke:
            assert _activate(agent) is True
            route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "hi"}], turn_id="t1")
            result = {"final_response": "default copy"}
            classified = SimpleNamespace(reason=FailoverReason.rate_limit, retryable=True, error_context={})
            assert route_hooks.routes_exhausted(agent, dict(result), classified=classified,
                                                provider="openai", model="gpt-4o") == result
        invoke.assert_not_called()
        assert (agent.provider, agent.model, agent._fallback_index) == ("openai", "gpt-4o", 1)

    def test_required_entry_is_skipped_without_a_custody_plugin(self, hooks):
        agent = _make_agent([{**HOSTED, "route_custody": "required"}, LOCAL])
        assert _activate(agent) is True
        assert (agent.provider, agent.model) == ("zai", "glm-4.7")


# ── F2: turn_route ─────────────────────────────────────────────────────────────────────


class TestTurnRoute:
    def test_turn_labels_grow_for_the_session_and_reset_with_it(self, hooks):
        seen = []

        def turn_route(phase, data_use, labels, messages, **kw):
            seen.append((phase, data_use, labels, len(messages), kw["turn_id"]))
            return {"labels": ["acme:client"]} if "acme:client-raw" in data_use else {"labels": "acme:seen"}

        hooks["turn_route"] = [turn_route]
        agent = _make_agent([HOSTED])
        messages = [{"role": "user", "content": "q"}, _tool_message(["acme:client-raw"])]
        route_hooks.begin_turn(agent, messages=messages, turn_id="t1")
        route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "q2"}], turn_id="t2")
        assert seen[0] == ("turn", ["acme:client-raw"], [], 2, "t1")
        assert seen[1][2] == ["acme:client"]  # labels from the first turn reach the second
        assert route_hooks._session_labels(agent) == {"acme:client", "acme:seen"}
        agent.session_id = "another-session"
        assert route_hooks._session_labels(agent) == frozenset()

    def test_a_raising_turn_route_adds_no_labels(self, hooks):
        def broken(**_kw):
            raise RuntimeError("classifier down")

        hooks["turn_route"] = [broken]
        agent = _make_agent([HOSTED])
        route_hooks.begin_turn(agent, messages=[], turn_id="t1")
        assert route_hooks._session_labels(agent) == frozenset()


# ── F3: route_custody ──────────────────────────────────────────────────────────────────


class TestRouteCustody:
    def test_skip_is_per_turn_and_never_sees_the_entry_secrets(self, hooks):
        calls = []

        def custody(candidate, labels, data_use, **kw):
            calls.append((dict(candidate), kw))
            return {"action": "skip", "reason": "client data"} if candidate["provider"] == "openai" else None

        hooks["route_custody"] = [custody]
        agent = _make_agent([HOSTED, LOCAL])
        route_hooks.begin_turn(agent, messages=[_tool_message(["acme:client-raw"])], turn_id="t1")
        assert _activate(agent) is True
        assert (agent.provider, agent.model) == ("zai", "glm-4.7")
        assert calls[0][0] == {"provider": "openai", "model": "gpt-4o"}
        assert all("fb-secret" not in repr(c) for c in calls)
        # Not marked unavailable: a later turn asks the gate again.
        assert ("openai", "gpt-4o") not in {k[:2] for k in agent._unavailable_fallback_keys}

    @pytest.mark.parametrize("failure", ["raise", "timeout"])
    def test_a_failing_gate_refuses(self, hooks, monkeypatch, failure):
        monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1)
        hang = threading.Event()

        def custody(candidate, **_kw):
            if candidate["provider"] == "zai":
                return {"action": "allow"}
            if failure == "raise":
                raise RuntimeError("classifier crashed")
            hang.wait(timeout=10.0)  # a stand-in gate that never answers

        hooks["route_custody"] = [custody]
        agent = _make_agent([HOSTED, LOCAL])
        primary = agent.provider
        try:
            if failure == "raise":
                assert _activate(agent) is True and agent.provider == "zai"
            else:
                # A hung gate is suppressed after its timeout, and a suppressed gate refuses every
                # candidate, so the turn stays on the primary instead of reaching a fallback.
                assert _activate(agent) is False and agent.provider == primary
        finally:
            hang.set()

    def test_required_entry_needs_an_explicit_allow(self, hooks):
        hooks["route_custody"] = [lambda **_kw: None]
        agent = _make_agent([{**HOSTED, "route_custody": "required"}, LOCAL])
        assert _activate(agent) is True and agent.provider == "zai"
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        agent = _make_agent([{**HOSTED, "route_custody": "required"}, LOCAL])
        assert _activate(agent) is True and agent.provider == "openai"

    def test_any_refusal_wins_over_an_allow(self):
        assert route_hooks.custody_verdict([{"action": "allow"}, "deny"], required=True)[0] is True
        assert route_hooks.custody_verdict([{"action": "block", "message": "m"}], required=False) == (True, "m")
        assert route_hooks.custody_verdict(["allow"], required=True) == (False, "")
        assert route_hooks.custody_verdict([None, 3], required=False) == (False, "")

    def test_a_kept_fallback_refused_at_turn_start_restores_the_primary(self, hooks):
        verdict = {"action": "allow"}
        hooks["route_custody"] = [lambda **_kw: verdict]
        agent = _make_agent([HOSTED])
        assert _activate(agent) is True and agent._provider_fallback_active
        agent._rate_limited_until = time.monotonic() + 600  # primary cooling down: the fallback is kept
        restore = MagicMock(return_value=True)
        agent._restore_primary_runtime = restore
        route_hooks.begin_turn(agent, messages=[], turn_id="t2")
        restore.assert_not_called()  # allowed: the kept fallback carries the turn
        verdict = {"action": "skip"}
        hooks["route_custody"] = [lambda **_kw: verdict]
        route_hooks.begin_turn(agent, messages=[], turn_id="t3")
        restore.assert_called_once()
        assert agent._rate_limited_until > time.monotonic()  # cooldown kept for the next failure

    def test_a_refused_fallback_with_no_primary_stops_the_turn(self, hooks):
        agent = _make_agent([HOSTED, LOCAL])
        assert _activate(agent) is True and agent.provider == "openai"
        hooks["route_custody"] = [lambda **_kw: {"action": "skip"}]
        agent._restore_primary_runtime = MagicMock(return_value=False)
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[], turn_id="t2")


# ── F2 and F3 at resolution time ───────────────────────────────────────────────────────


class TestResolutionTime:
    CFG = {"fallback_providers": [{**HOSTED, "route_custody": "required"}, LOCAL]}

    @staticmethod
    def _resolver(**kw):
        if kw.get("requested") == "anthropic":
            raise AuthError("primary quota exhausted")
        return {"provider": kw["requested"], "api_key": "k"}

    def test_required_entry_is_skipped_with_no_plugin(self, hooks, monkeypatch):
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", self._resolver)
        runtime, entry = resolve_runtime_with_fallback(self.CFG, requested="anthropic", target_model="opus")
        assert (runtime["provider"], entry["model"]) == ("zai", "glm-4.7")

    def test_gate_sees_resolve_phase_labels_and_no_messages(self, hooks, monkeypatch):
        seen = {}
        def turn_route(phase, chain, **_kw):
            seen["turn_route"] = (phase, chain)
            return {"labels": ["unclassified"]}

        hooks["turn_route"] = [turn_route]

        def custody(phase, candidate, labels, messages, data_use, **_kw):
            seen.setdefault("custody", []).append((phase, candidate["provider"], labels, messages, data_use))
            return {"action": "skip"} if "unclassified" in labels and candidate["provider"] == "openai" else None

        hooks["route_custody"] = [custody]
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", self._resolver)
        cfg = {"fallback_providers": [HOSTED, LOCAL]}
        runtime, _entry = resolve_runtime_with_fallback(cfg, requested="anthropic", target_model="opus")
        assert runtime["provider"] == "zai"
        assert seen["turn_route"] == ("resolve", [{"provider": "openai", "model": "gpt-4o"},
                                                  {"provider": "zai", "model": "glm-4.7"}])
        assert seen["custody"][0] == ("resolve", "openai", ["unclassified"], [], [])


# ── F6: on_routes_exhausted ────────────────────────────────────────────────────────────


class TestRoutesExhausted:
    CLASSIFIED = SimpleNamespace(reason=FailoverReason.rate_limit, retryable=True,
                                 error_context={"reset_at": time.time() + 3600})

    def _exhausted_agent(self):
        agent = _make_agent([HOSTED])
        assert _activate(agent) is True
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False
        return agent

    def test_reply_replaces_the_exhausted_copy(self, hooks):
        events = []

        def exhausted(**kw):
            events.append(kw)
            return {"reply": "  Back at 14:05.  "}

        hooks["on_routes_exhausted"] = [exhausted]
        result = route_hooks.routes_exhausted(
            self._exhausted_agent(), {"final_response": "Rate limited after 3 retries", "error": "429"},
            classified=self.CLASSIFIED, provider="openai", model="gpt-4o")
        assert result["final_response"] == "Back at 14:05."
        assert result["routes_exhausted"] is True and result["routes_exhausted_reply"] is True
        assert result["error"] == "429"
        event = events[0]
        assert event["reason"] == "rate_limit" and 3500 < event["reset_in_seconds"] <= 3600
        assert event["route"] == {"provider": "openai", "model": "gpt-4o"}
        assert "messages" not in event  # values-free: routes, labels and timing only

    def test_no_reply_keeps_the_copy_and_still_marks_the_event(self, hooks):
        hooks["on_routes_exhausted"] = [lambda **_kw: None]
        result = route_hooks.routes_exhausted(self._exhausted_agent(), {"final_response": "default"},
                                              classified=self.CLASSIFIED, provider="openai", model="gpt-4o")
        assert result == {"final_response": "default", "routes_exhausted": True}

    def test_not_fired_while_a_fallback_remains(self, hooks):
        fired = []
        hooks["on_routes_exhausted"] = [lambda **kw: fired.append(kw)]
        agent = _make_agent([HOSTED, LOCAL])
        result = route_hooks.routes_exhausted(agent, {"final_response": "default"},
                                              classified=self.CLASSIFIED, provider="x", model="y")
        assert result == {"final_response": "default"} and fired == []

    def test_a_raising_callback_keeps_the_default_copy(self, hooks):
        def broken(**_kw):
            raise RuntimeError("pager down")

        hooks["on_routes_exhausted"] = [broken]
        result = route_hooks.routes_exhausted(self._exhausted_agent(), {"final_response": "default"},
                                              classified=self.CLASSIFIED, provider="openai", model="gpt-4o")
        assert result["final_response"] == "default"


# ── Review fixes: custody on every path that can reach a required entry ─────────────────


def _client_for(provider=None, model=None, **_kw):
    client = MagicMock()
    client.base_url, client.api_key = f"https://{provider}.example/v1", "k"
    return client, model


class TestCustodyValue:
    @pytest.mark.parametrize("value", ["required", " Required ", True, "yes", "true", 1, "requierd"])
    def test_required_or_unrecognised_values_gate_the_entry(self, value):
        assert route_hooks.entry_requires_custody({**HOSTED, "route_custody": value}) is True

    @pytest.mark.parametrize("value", [None, False, "", "optional", "off", "none", "no", "false"])
    def test_off_values_leave_the_entry_ungated(self, value):
        assert route_hooks.entry_requires_custody({**HOSTED, "route_custody": value}) is False

    def test_an_unrecognised_value_warns_and_fails_closed(self, hooks, caplog):
        route_hooks._warned_values.discard("yes")
        agent = _make_agent([{**HOSTED, "route_custody": "yes"}, LOCAL])
        with caplog.at_level("WARNING", logger="agent.route_hooks"):
            assert _activate(agent) is True
        assert agent.provider == "zai"
        assert "unrecognised route_custody value" in caplog.text


class TestAuxiliaryCustody:
    CHAIN = [{**HOSTED, "route_custody": "required"}, LOCAL]

    def test_auto_auxiliary_call_skips_a_required_main_chain_entry(self, hooks):
        """A provider: auto auxiliary call whose main route fails walks fallback_providers; the
        required entry carries none of it while no gate is loaded."""
        resolved = []

        def resolve_entry(entry):
            resolved.append(entry["provider"])
            return _client_for(entry["provider"], entry["model"])

        with (
            patch("agent.auxiliary_client._try_main_provider_route", return_value=None),
            patch("hermes_cli.config.load_config_readonly", return_value={"fallback_providers": self.CHAIN}),
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={}),
            patch("agent.auxiliary_client._resolve_fallback_entry", side_effect=resolve_entry),
            patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False),
            patch("agent.auxiliary_client._context_too_small", return_value=None),
        ):
            from agent.auxiliary_client import _resolve_auto_route
            _client, model, provider = _resolve_auto_route({"provider": "anthropic", "model": "opus"}, task="compression")
        assert (provider, model) == ("zai", "glm-4.7")
        assert resolved == ["zai"]

    def test_task_chain_and_gate_see_the_task_and_session_labels(self, hooks):
        seen = []

        def custody(phase, candidate, task, labels, **_kw):
            seen.append((phase, candidate["provider"], task, labels))
            return {"action": "skip"} if candidate["provider"] == "openai" else None

        hooks["route_custody"] = [custody]
        route_hooks._remember_session_labels("aux-session", frozenset({"acme:client"}))
        from hermes_logging import clear_session_context, set_session_context
        set_session_context("aux-session")
        try:
            with (
                patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": [HOSTED, LOCAL]}),
                patch("agent.auxiliary_client._resolve_fallback_entry", side_effect=lambda e: _client_for(e["provider"], e["model"])),
                patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False),
            ):
                from agent.auxiliary_client import _try_configured_fallback_chain
                _client, model, label = _try_configured_fallback_chain("compression", "anthropic", reason="rate limit")
        finally:
            clear_session_context()
        assert model == "glm-4.7" and "zai" in label
        assert seen[0] == ("auxiliary", "openai", "compression", ["acme:client"])

    def test_compression_stall_retry_route_skips_a_required_entry(self, hooks):
        with (
            patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": self.CHAIN}),
            patch("agent.auxiliary_client._fallback_entry_api_key", return_value="k"),
        ):
            from agent.conversation_compression import resolve_compression_fallback_route
            route = resolve_compression_fallback_route()
        assert (route["provider"], route["model"]) == ("zai", "glm-4.7")


class _KeptFallbackAgent:
    """Stand-in for an agent kept on a fallback while its primary cools down. Records the route
    each turn-start compression request would have gone to."""

    def __new__(cls, *, restorable):
        from tests.agent.test_turn_context import _FakeAgent

        agent = _FakeAgent()
        agent.provider, agent.model = "openai", "gpt-4o"
        agent._fallback_chain, agent._fallback_index = [HOSTED], 1
        agent._provider_fallback_active, agent._route_fallback_entry = True, dict(HOSTED)
        agent._rate_limited_until = time.monotonic() + 600
        agent.compression_routes = []

        def restore():
            if agent._rate_limited_until > time.monotonic() or not restorable:
                return False
            agent.provider, agent.model, agent._provider_fallback_active = "anthropic", "opus", False
            return True

        def compress(messages, *_a, **_kw):
            agent.compression_routes.append((agent.provider, agent.model))
            return messages, "SYSTEM"

        agent._restore_primary_runtime = restore
        agent._compress_context = compress
        agent.compression_enabled = True
        agent.max_compression_attempts = 3
        agent.context_compressor = SimpleNamespace(
            protect_first_n=2, protect_last_n=2, threshold_tokens=1_000, context_length=200_000,
            summary_target_ratio=0.3, last_prompt_tokens=0, should_compress=lambda tokens=None: True,
            should_compress_info=lambda tokens=None: (True, None),
            get_active_compression_failure_cooldown=lambda: None)
        return agent


class TestTurnStartOrder:
    HISTORY = [{"role": "user", "content": "client rows"}, {"role": "assistant", "content": "x" * 8_000}]

    @pytest.fixture(autouse=True)
    def _quiet_runtime_main(self):
        with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
            yield

    def _turn(self, agent):
        from tests.agent.test_turn_context import _build
        return _build(agent, conversation_history=list(self.HISTORY))

    def test_refused_kept_fallback_never_receives_turn_start_compaction(self, hooks):
        hooks["route_custody"] = [lambda candidate, **_kw: {"action": "skip"} if candidate["provider"] == "openai" else None]
        agent = _KeptFallbackAgent(restorable=False)
        with pytest.raises(route_hooks.RouteCustodyRefused):
            self._turn(agent)
        assert agent.compression_routes == []

    def test_compaction_runs_on_the_restored_primary(self, hooks):
        hooks["route_custody"] = [lambda candidate, **_kw: {"action": "skip"} if candidate["provider"] == "openai" else None]
        agent = _KeptFallbackAgent(restorable=True)
        self._turn(agent)
        assert agent.compression_routes and set(agent.compression_routes) == {("anthropic", "opus")}
        assert agent._rate_limited_until > time.monotonic()  # the primary keeps its cooldown


class TestResolutionTimeAgent:
    """An agent built on a fallback before it existed (gateway, one-shot, TUI, cron): its active
    route is an entry of its own chain and nothing marks a fallback as active."""

    @staticmethod
    def _built_on(entry, chain, session_id="s1"):
        agent = _make_agent(chain)
        agent.provider, agent.model, agent.session_id = entry["provider"], entry["model"], session_id
        assert not getattr(agent, "_provider_fallback_active", False)
        return agent

    def test_required_entry_stops_the_turn_without_a_gate(self, hooks):
        required = {**HOSTED, "route_custody": "required"}
        agent = self._built_on(required, [required, LOCAL])
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "hi"}], turn_id="t1")

    def test_plain_entry_is_untouched_without_a_gate(self, hooks):
        agent = self._built_on(HOSTED, [HOSTED, LOCAL])
        with patch.object(route_hooks, "_invoke") as invoke:
            route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "hi"}], turn_id="t1")
        invoke.assert_not_called()

    def test_gate_sees_the_turn_and_earlier_session_labels(self, hooks):
        seen = []
        hooks["turn_route"] = [lambda data_use, **_kw: {"labels": ["acme:client"]} if data_use else None]

        def custody(phase, candidate, labels, data_use, messages, **_kw):
            seen.append((phase, candidate["provider"], labels, data_use, len(messages)))
            return {"action": "skip"} if "acme:client" in labels and candidate["provider"] == "openai" else None

        hooks["route_custody"] = [custody]
        first = _make_agent([HOSTED, LOCAL])
        first.session_id = "gw-session"
        route_hooks.begin_turn(first, messages=[_tool_message(["acme:client-raw"])], turn_id="t1")
        # The primary then fails; the gateway builds a new agent for the same session on HOSTED.
        rebuilt = self._built_on(HOSTED, [HOSTED, LOCAL], session_id="gw-session")
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(rebuilt, messages=[{"role": "user", "content": "plain"}], turn_id="t2")
        assert seen[-1] == ("turn", "openai", ["acme:client"], [], 1)

    def test_resolve_phase_gate_sees_labels_of_a_named_session_only(self, hooks):
        seen = []
        hooks["route_custody"] = [lambda labels, session_id, **_kw: seen.append((labels, session_id))]
        route_hooks._remember_session_labels("gw-2", frozenset({"acme:client"}))
        primary = {"provider": "anthropic", "model": "opus"}
        route_hooks.resolution_refuses(HOSTED, primary=primary, chain=[HOSTED], labels_cache={}, session_id="gw-2")
        from hermes_logging import clear_session_context, set_session_context
        set_session_context("gw-2")  # a stale thread binding is not trusted at resolve time
        try:
            route_hooks.resolution_refuses(HOSTED, primary=primary, chain=[HOSTED], labels_cache={})
        finally:
            clear_session_context()
        assert seen == [(["acme:client"], "gw-2"), ([], "")]

    def test_refusal_ends_the_turn_with_the_routes_exhausted_reply(self, hooks):
        events = []
        hooks["on_routes_exhausted"] = [lambda **kw: events.append(kw) or {"reply": "Holding: no cleared route."}]
        agent = self._built_on(HOSTED, [HOSTED])
        result = route_hooks.route_custody_refused_result(
            agent, route_hooks.RouteCustodyRefused("refused"), [{"role": "user", "content": "earlier"}])
        assert result["final_response"] == "Holding: no cleared route."
        assert (result["failure_reason"], result["api_calls"], result["failed"]) == ("route_custody", 0, True)
        assert result["messages"] == [{"role": "user", "content": "earlier"}]
        assert events[0]["reason"] == "route_custody" and "messages" not in events[0]

    def test_conversation_loop_returns_the_refusal_result(self, hooks):
        agent = _make_agent([HOSTED])
        with patch("agent.conversation_loop.build_turn_context",
                   side_effect=route_hooks.RouteCustodyRefused("refused")):
            result = agent.run_conversation("hi")
        assert result["failure_reason"] == "route_custody" and result["api_calls"] == 0

    def test_init_time_walker_skips_a_required_entry(self, hooks):
        from agent.agent_init import _routed_client_kwargs

        def resolve(provider, model=None, **_kw):
            return (None, None) if provider == "anthropic" else _client_for(provider, model)

        agent = SimpleNamespace(provider="anthropic", requested_provider="anthropic", model="opus",
                                api_mode="chat_completions", _fallback_activated=False)
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve):
            kwargs = _routed_client_kwargs(agent, [{**HOSTED, "route_custody": "required"}, LOCAL], None)
        assert (agent.provider, agent.model) == ("zai", "glm-4.7")
        assert kwargs["base_url"] == "https://zai.example/v1"


class TestCronCustody:
    def test_cron_walker_skips_a_required_entry_with_no_plugin(self, hooks):
        from cron.scheduler import _CronJobConfig, _resolve_job_runtime

        chain = [{"provider": "openai", "model": "gpt-4o", "route_custody": "required"},
                 {"provider": "zai", "model": "glm-4.7"}]
        jc = _CronJobConfig(cfg={"fallback_providers": chain}, model="opus", model_cfg={}, cron_default_provider="")

        def resolve(**kw):
            if kw.get("requested") in (None, "anthropic"):
                raise AuthError("No credentials stored")
            return {"api_key": "k", "base_url": "https://x.example/v1", "provider": kw["requested"],
                    "api_mode": "chat_completions"}

        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve):
            runtime, model = _resolve_job_runtime({"id": "j1", "prompt": "p"}, "j1", jc)
        assert (runtime["provider"], model) == ("zai", "glm-4.7")


class TestRoutesExhaustedReplyScope:
    def _agent(self):
        agent = _make_agent([HOSTED])
        assert _activate(agent) is True
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False
        return agent

    def test_auth_copy_keeps_its_relogin_guidance(self, hooks):
        hooks["on_routes_exhausted"] = [lambda **_kw: {"reply": "busy"}]
        classified = SimpleNamespace(reason=FailoverReason.auth_permanent, retryable=False, error_context={})
        result = route_hooks.routes_exhausted(self._agent(), {"final_response": "Run hermes login"},
                                              classified=classified, provider="openai", model="gpt-4o")
        assert result["final_response"] == "Run hermes login" and result["routes_exhausted"] is True
        assert "routes_exhausted_reply" not in result

    def test_delivered_partial_text_stays_in_front_of_the_reply(self, hooks):
        hooks["on_routes_exhausted"] = [lambda **_kw: {"reply": "Back at 14:05."}]
        classified = SimpleNamespace(reason=FailoverReason.rate_limit, retryable=True, error_context={})
        result = route_hooks.routes_exhausted(
            self._agent(), {"final_response": "Half an answer\n\nRate limited", "partial": True},
            classified=classified, provider="openai", model="gpt-4o", delivered="Half an answer")
        assert result["final_response"] == "Half an answer\n\nBack at 14:05."

    def test_free_tier_card_keeps_its_copy(self, hooks):
        hooks["on_routes_exhausted"] = [lambda **_kw: {"reply": "busy"}]
        classified = SimpleNamespace(reason=FailoverReason.rate_limit, retryable=True, error_context={})
        result = route_hooks.routes_exhausted(
            self._agent(), {"final_response": "Sign in", "free_tier": {"kind": "refused", "message": "Sign in"}},
            classified=classified, provider="openai", model="gpt-4o")
        assert result["final_response"] == "Sign in"
