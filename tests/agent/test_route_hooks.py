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
