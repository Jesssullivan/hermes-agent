"""F12 on built-in discovery and on the model a vision slot sends (agent/route_hooks.py).

Text discovery judges the candidate it picked when ``model.route_custody`` covers it. A vision
slot is judged on the model it sends, so a gate that decides by model sees that model. An
availability probe asks no gate. Each probe drives the production code with a stub client. No
provider is called."""

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux
from agent import route_hooks
from hermes_cli.plugins import PluginManager


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture(autouse=True)
def _no_scope():
    token = route_hooks._ROUTE_SCOPE.set(None)
    yield
    route_hooks._ROUTE_SCOPE.reset(token)


def _mark(provider):
    return patch("agent.route_hooks._main_route_marker", return_value=(provider, True))


def _gate(asked, refuse=lambda provider, model: False):
    """Records ``(provider, model)`` for each candidate and refuses what *refuse* names."""
    def custody(candidate, **_kw):
        asked.append((candidate["provider"], candidate["model"]))
        return {"action": "skip"} if refuse(candidate["provider"], candidate["model"]) else {"action": "allow"}
    return custody


def _discover(chain, task=None):
    """``_resolve_auto_route`` on an install with no main provider, so the walk reaches discovery."""
    with patch.object(aux, "_main_route_target", return_value=("", "", "", "", "")), \
            patch.object(aux, "_warn_stale_openai_base_url"), \
            patch.object(aux, "_normalize_main_runtime", return_value={}), \
            patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")), \
            patch.object(aux, "_is_provider_unhealthy", return_value=False), \
            patch.object(aux, "_custom_health_base_url", return_value=None), \
            patch.object(aux, "_get_provider_chain", return_value=chain):
        return aux._resolve_auto_route(task=task)


class TestTextDiscovery:
    def test_a_candidate_the_mark_covers_is_refused_when_the_gate_refuses_it(self, hooks):
        asked, client = [], MagicMock()
        hooks["route_custody"] = [_gate(asked, refuse=lambda provider, _m: provider == "openrouter")]
        with _mark(""):
            result = _discover([("openrouter", lambda: (client, "or-model"))])
        assert result == (None, None, "")
        assert asked == [("", ""), ("openrouter", "or-model")]

    def test_a_candidate_the_gate_admits_is_used(self, hooks):
        asked, client = [], MagicMock()
        hooks["route_custody"] = [_gate(asked)]
        with _mark(""):
            assert _discover([("openrouter", lambda: (client, "or-model"))]) == (client, "or-model", "openrouter")
        assert ("openrouter", "or-model") in asked

    def test_with_no_gate_a_covered_candidate_carries_nothing(self, hooks):
        with _mark(""):
            assert _discover([("nous", lambda: (MagicMock(), "n-model"))]) == (None, None, "")

    def test_local_custom_is_judged_as_custom(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda provider, _m: provider == "custom")]
        with _mark(""):
            assert _discover([("local/custom", lambda: (MagicMock(), "local-m"))])[0] is None
        assert ("custom", "local-m") in asked

    def test_the_api_key_candidate_names_no_route_and_is_refused_unasked(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        with _mark(""):
            assert _discover([("api-key", lambda: (MagicMock(), "k-model"))])[0] is None
        assert ("api-key", "k-model") not in asked and all(row[0] == "" for row in asked)

    def test_with_no_mark_discovery_is_unchanged(self, hooks):
        client = MagicMock()
        with patch("agent.route_hooks._main_route_marker", return_value=("", False)):
            assert _discover([("api-key", lambda: (client, "k-model"))]) == (client, "k-model", "api-key")


class TestVisionModelInUse:
    def test_the_main_route_is_judged_on_the_vision_model_it_sends(self, hooks):
        asked, client = [], MagicMock()
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"), patch.object(aux, "_resolve_provider_vision_default", return_value="vision-x"), \
                patch.object(aux, "_main_model_supports_vision", return_value=True), \
                patch.object(aux, "resolve_provider_client", side_effect=lambda _p, model, **_kw: (client, model)):
            provider, sent_client, sent = aux._vision_auto_route({"provider": "zai", "model": "text-m"}, None, None, False)
        assert (provider, sent_client, sent) == ("zai", client, "vision-x")
        assert asked == [("zai", "vision-x")]

    def test_a_gate_refusing_the_vision_model_stops_the_main_route(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "vision-x")]
        with _mark("zai"), patch.object(aux, "_resolve_provider_vision_default", return_value="vision-x"), \
                patch.object(aux, "_vision_main_provider_client") as ride, \
                patch.object(aux, "_resolve_strict_vision_backend") as discovery:
            assert aux._vision_auto_route({"provider": "zai", "model": "text-m"}, None, None, False) == (None, None, None)
        ride.assert_not_called()
        discovery.assert_not_called()

    def test_a_model_the_backend_picks_is_judged_before_it_is_returned(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "nous-v")]
        with _mark("nous"), patch.object(aux, "_resolve_provider_vision_default", return_value=None), \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(MagicMock(), "nous-v")) as strict:
            assert aux._vision_auto_route({"provider": "nous", "model": "chat-m"}, None, None, False)[1] is None
        strict.assert_called_once()
        assert asked == [("nous", ""), ("nous", "nous-v")]

    def test_a_discovery_candidate_is_judged_on_the_model_it_picked(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda provider, model: (provider, model) == ("openrouter", "or-v"))]
        with _mark(""), patch.object(aux, "_vision_main_provider_client", return_value=(None, None)), \
                patch.object(aux, "_resolve_provider_vision_default", return_value=None), \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(MagicMock(), "or-v")) as strict:
            assert aux._vision_auto_route({"provider": "zai", "model": "m"}, None, None, False)[1] is None
        strict.assert_called_once()
        assert ("openrouter", "or-v") in asked

    def test_an_admitted_model_is_asked_once(self, hooks):
        asked, client = [], MagicMock()
        hooks["route_custody"] = [_gate(asked)]
        with _mark("openrouter"), \
                patch.object(aux, "_resolve_task_provider_model", return_value=("openrouter", "or-v", None, None, None)), \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(client, "or-v")):
            assert aux.resolve_vision_provider_client()[1:] == (client, "or-v")
        assert asked == [("openrouter", "or-v")]

    def test_a_slot_with_no_model_is_judged_on_the_default_it_sends(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "or-default")]
        with _mark("openrouter"), \
                patch.object(aux, "_resolve_task_provider_model", return_value=("openrouter", None, None, None, None)), \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(MagicMock(), "or-default")):
            assert aux.resolve_vision_provider_client()[1] is None
        assert asked == [("openrouter", ""), ("openrouter", "or-default")]

    def test_a_base_url_slot_is_judged_on_the_model_it_sends(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "or-default")]
        with _mark("openrouter"), \
                patch.object(aux, "_resolve_task_provider_model", return_value=("openrouter", None, "https://u/v1", "k", None)), \
                patch.object(aux, "resolve_provider_client", return_value=(MagicMock(), "or-default")):
            assert aux.resolve_vision_provider_client()[1] is None
        assert asked[-1] == ("openrouter", "or-default")


class TestMarkedEndpoint:
    def test_a_custom_slot_at_the_marked_endpoint_is_judged_as_the_marked_route(self, hooks):
        """The mark compares endpoints too: a ``custom`` slot whose base_url has the marked
        provider's scheme, host and port is judged as that provider
        (tests/agent/test_route_hooks_endpoint_compare.py)."""
        asked, client = [], MagicMock()
        hooks["route_custody"] = [_gate(asked)]
        with _mark("openrouter"), patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=None), \
                patch.object(aux, "_resolve_task_provider_model",
                             return_value=("custom", "m", "https://openrouter.ai/api/v1", "k", None)), \
                patch.object(aux, "resolve_provider_client", return_value=(client, "m")):
            assert aux.resolve_vision_provider_client()[1] is client
        assert asked == [("openrouter", "m")]


class TestAvailabilityProbe:
    def test_a_text_probe_asks_no_gate(self, hooks):
        asked, stub = [], MagicMock()
        hooks["route_custody"] = [_gate(asked, refuse=lambda *_a: True)]
        with _mark("openrouter"), aux.aux_probe_mode(), patch.dict(aux._client_cache, clear=True), \
                patch.object(aux, "resolve_provider_client", return_value=(stub, "m")):
            assert aux._get_cached_client("openrouter", "m", task="compression")[0] is stub
        assert asked == []

    def test_a_discovery_probe_asks_no_gate(self, hooks):
        asked, stub = [], MagicMock()
        hooks["route_custody"] = [_gate(asked, refuse=lambda *_a: True)]
        with _mark(""), aux.aux_probe_mode():
            assert _discover([("openrouter", lambda: (stub, "or-model"))])[0] is stub
        assert asked == []

    def test_outside_a_probe_the_same_call_is_refused(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda *_a: True)]
        with _mark("openrouter"), patch.object(aux, "resolve_provider_client") as build:
            assert aux._get_cached_client("openrouter", "m", task="compression") == (None, None)
        build.assert_not_called()
        assert asked == [("openrouter", "m")]
