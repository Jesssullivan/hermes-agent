"""F12 on a slot's own model and on a slot's ``base_url`` under a marked ``model`` block
(agent/route_hooks.py).

An ``auto`` text slot that names its own model is judged on that model, on the route the walk
picked. A slot or discovery candidate whose ``base_url`` has the marked route's scheme, host and
port is judged as the marked route, and so is a built client that sends there. When the marked
route's endpoint cannot be resolved, every slot with a ``base_url`` is judged as the marked route.
Each probe drives the production code with a stub client. No provider is called."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux
from agent import route_hooks
from hermes_cli.config_read_errors import FailedConfigRead
from hermes_cli.plugins import PluginManager

ZAI = "https://api.z.ai/api/paas/v4"


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


@pytest.fixture(autouse=True)
def _no_runtime(monkeypatch):
    monkeypatch.setattr(aux, "_normalize_main_runtime", lambda _runtime=None: {})


def _mark(provider):
    return patch("agent.route_hooks._main_route_marker", return_value=(provider, True))


def _config(block):
    return patch("hermes_cli.config.read_raw_config_readonly", return_value={"model": block})


def _gate(asked, refuse=lambda provider, model: False):
    """Records ``(provider, model)`` for each candidate and refuses what *refuse* names."""
    def custody(candidate, **_kw):
        asked.append((candidate["provider"], candidate["model"]))
        return {"action": "skip"} if refuse(candidate["provider"], candidate["model"]) else {"action": "allow"}
    return custody


def _client(base_url=None):
    client = MagicMock()
    if base_url is not None:
        client.base_url = base_url
    return client


def _auto_slot(main, chain=(), model="slot-m"):
    """``resolve_provider_client("auto", model=...)`` with a main route of *main* ``(provider, model)``."""
    provider, main_model = main
    routed = (_client(), main_model, provider) if provider else None
    with patch.object(aux, "_main_route_target", return_value=(provider, main_model, "", "", "")), \
            patch.object(aux, "_try_main_provider_route", return_value=routed), \
            patch.object(aux, "_warn_stale_openai_base_url"), \
            patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")), \
            patch.object(aux, "_is_provider_unhealthy", return_value=False), \
            patch.object(aux, "_custom_health_base_url", return_value=None), \
            patch.object(aux, "_get_provider_chain", return_value=list(chain)):
        return aux.resolve_provider_client("auto", model=model, task="compression")


def _no_named():
    return patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=None)


class TestAutoSlotModel:
    def test_step_one_is_judged_on_the_model_the_slot_sends(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "slot-m")]
        with _mark("zai"):
            assert _auto_slot(("zai", "main-m")) == (None, None)
        assert asked == [("zai", "main-m"), ("zai", "slot-m")]

    def test_step_one_sends_the_slot_model_when_the_gate_admits_it(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"):
            client, model = _auto_slot(("zai", "main-m"))
        assert client is not None and model == "slot-m"
        assert ("zai", "slot-m") in asked

    def test_discovery_is_judged_on_the_model_the_slot_sends(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "slot-m")]
        with _mark(""):
            assert _auto_slot(("", ""), chain=[("zai", lambda: (_client(), "disc-m"))]) == (None, None)
        assert asked == [("", ""), ("zai", "disc-m"), ("zai", "slot-m")]

    def test_a_slot_with_no_model_is_judged_once(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"):
            assert _auto_slot(("zai", "main-m"), model=None)[1] == "main-m"
        assert asked == [("zai", "main-m")]

    def test_an_openrouter_format_model_the_walk_drops_is_not_judged(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "vendor/slot-m")]
        with _mark("zai"):
            client, model = _auto_slot(("zai", "main-m"), model="vendor/slot-m")
        assert client is not None and model == "main-m"
        assert asked == [("zai", "main-m")]

    def test_with_no_mark_the_slot_model_is_sent_unasked(self, hooks):
        with patch("agent.route_hooks._main_route_marker", return_value=("", False)):
            assert _auto_slot(("zai", "main-m"))[1] == "slot-m"


class TestEndpointResolved:
    """Option A, the marked route's endpoint resolves: compare scheme, host and port."""

    @pytest.mark.parametrize("url", [ZAI, "https://api.z.ai:443/other", "https://API.Z.AI/v1/"])
    def test_a_custom_slot_at_the_marked_endpoint_is_judged_as_the_marked_route(self, hooks, url):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda provider, _m: provider == "zai")]
        with _mark("zai"), _config({"provider": "zai"}):
            assert route_hooks.marked_main_refuses("custom", "glm-x", task="compression", base_url=url)
        assert asked == [("zai", "glm-x")]

    @pytest.mark.parametrize("url", ["http://api.z.ai/api/paas/v4", "https://api.z.ai:8443/api/paas/v4",
                                     "https://local.example:8080/v1"])
    def test_another_scheme_port_or_host_is_not_judged(self, hooks, url):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"), _config({"provider": "zai"}):
            assert not route_hooks.marked_main_refuses("custom", "glm-x", task="compression", base_url=url)
        assert asked == []

    def test_the_blocks_own_base_url_is_the_marked_endpoint(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, _m: True)]
        with _mark("openrouter"), _config({"provider": "openrouter", "base_url": "https://relay.example:8443/v1"}):
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url="https://relay.example:8443")
            assert not route_hooks.marked_main_refuses("custom", "m", task="title", base_url="https://relay.example")
        assert asked == [("openrouter", "m")]

    def test_the_bound_main_runtime_endpoint_is_the_marked_endpoint(self, hooks, monkeypatch):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, _m: True)]
        monkeypatch.setattr(aux, "_normalize_main_runtime",
                            lambda _runtime=None: {"provider": "zai", "base_url": "https://open.bigmodel.cn/api/paas/v4"})
        with _mark("zai"), _config({"provider": "zai"}):
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url="https://open.bigmodel.cn")
        assert asked == [("zai", "m")]

    def test_a_base_url_with_no_host_is_judged_as_the_marked_route(self, hooks):
        with _mark("zai"), _config({"provider": "zai"}):
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url="http:///v1")

    def test_the_registry_env_override_is_a_marked_endpoint(self, hooks, monkeypatch):
        monkeypatch.setenv("GLM_BASE_URL", "https://relay.example:9443/v1")
        with _mark("zai"), _config({"provider": "zai"}), patch("hermes_cli.providers.get_provider", return_value=None):
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url="https://relay.example:9443")

    def test_the_named_custom_providers_endpoint_is_a_marked_endpoint(self, hooks):
        named = {"base_url": "https://relay.example:7443/v1"}
        with _mark("relay"), _config({"provider": "relay", "base_url": "https://block.example/v1"}), \
                patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=named):
            assert route_hooks.marked_main_refuses("openrouter", "m", task="title", base_url="https://relay.example:7443")
            assert not route_hooks.marked_main_refuses("openrouter", "m", task="title", base_url="https://other.example")

    def test_the_catalog_endpoint_is_a_marked_endpoint(self, hooks, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", {})
        catalog = SimpleNamespace(base_url="https://catalog.example:6443/v1", base_url_env_var="")
        with _mark("zai"), _config({"provider": "zai", "base_url": "https://block.example/v1"}), _no_named(), \
                patch("hermes_cli.providers.get_provider", return_value=catalog):
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url="https://catalog.example:6443")
            assert not route_hooks.marked_main_refuses("custom", "m", task="title", base_url="https://other.example")

    def test_step_one_at_the_marked_endpoint_is_judged_as_the_marked_route(self, hooks):
        with _mark("zai"), _config({"provider": "zai"}), \
                patch.object(aux, "_main_route_target", return_value=("custom", "m", ZAI, "", "")), \
                patch.object(aux, "_try_main_provider_route") as step_one, \
                patch.object(aux, "_warn_stale_openai_base_url"), \
                patch.object(aux, "_try_configured_fallback_chain", return_value=(None, None, "")), \
                patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")), \
                patch.object(aux, "_get_provider_chain", return_value=[]):
            aux._resolve_auto_route(task="compression")
        step_one.assert_not_called()

    def test_with_no_gate_a_slot_at_the_marked_endpoint_carries_nothing(self, hooks):
        with _mark("zai"), _config({"provider": "zai"}):
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url=ZAI)
            assert not route_hooks.marked_main_refuses("custom", "m", task="title", base_url="http://127.0.0.1:8080/v1")

    def test_a_text_slot_at_the_marked_endpoint_gets_no_client(self, hooks):
        hooks["route_custody"] = [_gate([], refuse=lambda _p, _m: True)]
        slot = ("custom", "glm-x", ZAI, "", "")
        with _mark("zai"), _config({"provider": "zai"}), \
                patch.object(aux, "_resolve_task_provider_model", return_value=slot), \
                patch.object(aux, "resolve_provider_client") as build:
            assert aux.get_text_auxiliary_client("compression") == (None, None)
        build.assert_not_called()

    def test_a_cached_client_at_the_marked_endpoint_is_not_built(self, hooks):
        hooks["route_custody"] = [_gate([], refuse=lambda _p, _m: True)]
        with _mark("zai"), _config({"provider": "zai"}), patch.object(aux, "resolve_provider_client") as build:
            assert aux._get_cached_client("custom", "glm-x", base_url=ZAI, task="compression") == (None, None)
        build.assert_not_called()

    def test_a_vision_slot_at_the_marked_endpoint_sends_no_image(self, hooks):
        hooks["route_custody"] = [_gate([], refuse=lambda _p, _m: True)]
        with _mark("zai"), _config({"provider": "zai"}), patch.object(aux, "resolve_provider_client") as build:
            assert aux.resolve_vision_provider_client("custom", "v-m", base_url=ZAI)[1:] == (None, None)
        build.assert_not_called()

    def test_a_vision_override_is_judged_again_at_the_marked_endpoint(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, model: model == "v-final")]
        with _mark("zai"), _config({"provider": "zai"}), \
                patch.object(aux, "resolve_provider_client", return_value=(_client(), "v-final")):
            assert aux.resolve_vision_provider_client("custom", None, base_url=ZAI)[1:] == (None, None)
        assert asked == [("zai", ""), ("zai", "v-final")]

    def test_a_vision_backend_whose_client_is_at_the_marked_endpoint_sends_no_image(self, hooks):
        with _mark("zai"), _config({"provider": "zai"}), \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(_client(ZAI), "v-default")):
            assert aux.resolve_vision_provider_client("openrouter", None)[1:] == (None, None)

    def test_a_discovery_candidate_at_the_marked_endpoint_is_judged(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda provider, _m: provider == "zai")]
        with _mark("zai"), _config({"provider": "zai"}):
            result = _auto_slot(("", ""), chain=[("local/custom", lambda: (_client(ZAI), "local-m"))], model=None)
        assert result == (None, None)
        assert ("zai", "local-m") in asked

    def test_a_discovery_candidate_elsewhere_is_not_judged_by_the_mark(self, hooks):
        asked, local = [], _client("http://127.0.0.1:8080/v1")
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"), _config({"provider": "zai"}):
            client, model = _auto_slot(("", ""), chain=[("local/custom", lambda: (local, "local-m"))], model=None)
        assert client is not None and model == "local-m"
        assert asked == [("", "")]


class TestBuiltClient:
    """A slot with no configured ``base_url`` at the marked endpoint is judged on its built client's."""

    @pytest.fixture(autouse=True)
    def _empty_cache(self, monkeypatch):
        monkeypatch.setattr(aux, "_client_cache", {})

    @staticmethod
    def _text(slot, client):
        with patch.object(aux, "_resolve_task_provider_model", return_value=slot), \
                patch.object(aux, "resolve_provider_client", return_value=(client, slot[1])):
            return aux.get_text_auxiliary_client("compression")

    def test_a_text_slot_whose_client_reaches_the_marked_endpoint_carries_nothing(self, hooks):
        with _mark("zai"), _config({"provider": "zai"}):
            assert self._text(("custom", "glm-x", None, "", ""), _client(ZAI)) == (None, None)

    def test_a_text_slot_whose_client_is_elsewhere_is_sent(self, hooks):
        local = _client("http://127.0.0.1:8080/v1")
        with _mark("zai"), _config({"provider": "zai"}):
            assert self._text(("custom", "glm-x", None, "", ""), local) == (local, "glm-x")

    def test_the_gate_sees_the_marked_route_for_the_built_client(self, hooks):
        asked, built = [], _client(ZAI)
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"), _config({"provider": "zai"}):
            assert self._text(("custom", "glm-x", None, "", ""), built) == (built, "glm-x")
        assert asked == [("zai", "glm-x")]

    def test_a_slot_judged_before_the_build_is_not_asked_again(self, hooks):
        asked, built = [], _client(ZAI)
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"), _config({"provider": "zai"}):
            assert self._text(("custom", "glm-x", ZAI, "", ""), built) == (built, "glm-x")
        assert asked == [("zai", "glm-x")]

    def test_an_auto_slot_is_left_to_its_walk(self, hooks):
        asked, built = [], _client(ZAI)
        hooks["route_custody"] = [_gate(asked)]
        with _mark("zai"), _config({"provider": "zai"}):
            assert self._text(("auto", "glm-x", None, "", ""), built) == (built, "glm-x")
        assert asked == []

    def test_a_cached_client_at_the_marked_endpoint_is_judged_on_every_call(self, hooks):
        asked, refuse, built = [], [False], _client(ZAI)
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, _m: refuse[0])]
        with _mark("zai"), _config({"provider": "zai"}), \
                patch.object(aux, "resolve_provider_client", return_value=(built, "glm-x")) as build:
            assert aux._get_cached_client("custom", "glm-x", task="compression") == (built, "glm-x")
            refuse[0] = True
            assert aux._get_cached_client("custom", "glm-x", task="compression") == (None, None)
        assert build.call_count == 1
        assert asked == [("zai", "glm-x"), ("zai", "glm-x")]

    def test_a_named_vision_slot_whose_client_reaches_the_marked_endpoint_sends_no_image(self, hooks):
        with _mark("zai"), _config({"provider": "zai"}), \
                patch.object(aux, "resolve_provider_client", return_value=(_client(ZAI), "v-m")):
            assert aux.resolve_vision_provider_client("deepseek", "v-m")[1:] == (None, None)


class TestEndpointUnresolved:
    """Option A, the marked route's endpoint cannot be resolved: judge every slot with a ``base_url``."""

    @pytest.mark.parametrize("read", [
        {"side_effect": OSError("unreadable")},
        {"return_value": FailedConfigRead({}, error=OSError("unreadable"))},
        {"return_value": {"model": {"provider": "nowhere"}}},
    ])
    def test_every_base_url_slot_is_judged_as_the_marked_route(self, hooks, read):
        asked = []
        hooks["route_custody"] = [_gate(asked, refuse=lambda _p, _m: True)]
        with _mark("nowhere"), patch("hermes_cli.config.read_raw_config_readonly", **read), _no_named():
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url="http://127.0.0.1:8080/v1")
            assert route_hooks.marked_main_refuses("openrouter", "m", task="title", base_url="https://openrouter.ai/api/v1")
        assert asked == [("nowhere", "m"), ("nowhere", "m")]

    @pytest.mark.parametrize("read", [
        {"side_effect": OSError("unreadable")},
        {"return_value": {"model": {"provider": "nowhere"}}},
    ])
    @pytest.mark.parametrize("gate", [None, lambda **_kw: None, lambda **_kw: {"action": "observe"}])
    def test_a_slot_needs_an_allow_with_no_gate_or_no_verdict(self, hooks, read, gate):
        if gate is not None:
            hooks["route_custody"] = [gate]
        with _mark("nowhere"), patch("hermes_cli.config.read_raw_config_readonly", **read), _no_named():
            assert route_hooks.marked_main_refuses("custom", "m", task="title", base_url="http://127.0.0.1:8080/v1")
            assert route_hooks.marked_main_refuses("openrouter", "m", task="title", base_url="https://openrouter.ai/api/v1")

    def test_a_built_client_needs_an_allow(self, hooks):
        slot, built = ("custom", "m", None, "", ""), _client("http://127.0.0.1:8080/v1")
        with _mark("nowhere"), patch("hermes_cli.config.read_raw_config_readonly", side_effect=OSError("x")), _no_named(), \
                patch.object(aux, "_resolve_task_provider_model", return_value=slot), \
                patch.object(aux, "resolve_provider_client", return_value=(built, "m")):
            assert aux.get_text_auxiliary_client("title") == (None, None)

    def test_a_slot_with_no_base_url_is_still_read_by_name(self, hooks):
        asked = []
        hooks["route_custody"] = [_gate(asked)]
        with _mark("nowhere"), patch("hermes_cli.config.read_raw_config_readonly", side_effect=OSError("x")), \
                patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=None):
            assert not route_hooks.marked_main_refuses("openrouter", "m", task="title")
        assert asked == []

    def test_a_failing_named_lookup_leaves_the_endpoint_unresolved(self, hooks):
        with _mark("zai"), _config({"provider": "zai"}), \
                patch("hermes_cli.runtime_provider._get_named_custom_provider", side_effect=RuntimeError("x")):
            assert route_hooks._marked_route_origins() == frozenset()
            assert route_hooks.marked_main_refuses("openrouter", "m", task="title", base_url="http://127.0.0.1:8080/v1")
