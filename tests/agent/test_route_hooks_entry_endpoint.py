"""F12 at the marked endpoint under another name (agent/route_hooks.py).

A marked route's endpoint includes the URLs its key can select. A fallback chain entry is judged by
its URLs as well as its name, and so is the client it builds. A vision backend's built client is
judged on every path. An ``auto`` client is built again on each call while a chain it walks holds an
entry that needs an ``allow``. The gate sees the slot's own name as ``slot_provider``. Each probe
drives the production code with a stub client. No provider is called."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import auxiliary_client as aux
from agent import route_hooks
from hermes_cli.plugins import PluginManager

ZAI = "https://api.z.ai/api/paas/v4"
KIMI_CODE = "https://api.kimi.com/coding"
LOCAL = "http://127.0.0.1:8080/v1"


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: mgr)
    return mgr._hooks


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    token = route_hooks._ROUTE_SCOPE.set(None)
    monkeypatch.setattr(aux, "_normalize_main_runtime", lambda _runtime=None: {})
    monkeypatch.setattr(aux, "_client_cache", {})
    for name in ("KIMI_BASE_URL", "GLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    yield
    route_hooks._ROUTE_SCOPE.reset(token)


def _mark(provider, marked=True):
    return patch("agent.route_hooks._main_route_marker", return_value=(provider, marked))


def _config(block):
    return patch("hermes_cli.config.read_raw_config_readonly", return_value={"model": block})


def _named(table=None, fail=()):
    """``_get_named_custom_provider`` answering from *table*, and raising for a name in *fail*."""
    def lookup(name, **_kw):
        if name in fail:
            raise RuntimeError("unreadable providers entry")
        return (table or {}).get(name)
    return patch("hermes_cli.runtime_provider._get_named_custom_provider", side_effect=lookup)


def _gate(seen, refuse=lambda candidate: False):
    """Records each candidate and refuses what *refuse* names."""
    def custody(candidate, **_kw):
        seen.append(dict(candidate))
        return {"action": "skip"} if refuse(candidate) else {"action": "allow"}
    return custody


def _client(base_url=None):
    client = MagicMock()
    if base_url is not None:
        client.base_url = base_url
    return client


def _zai_mark():
    return _mark("zai"), _config({"provider": "zai"}), _named()


class TestKeySelectedEndpoints:
    """F-A: an ``sk-kimi-`` key and the Z.AI probe move a provider off its registry endpoint."""

    @pytest.mark.parametrize("url", [KIMI_CODE, KIMI_CODE + "/v1"])
    def test_a_slot_at_kimi_code_is_judged_as_the_kimi_mark(self, hooks, url):
        seen = []
        hooks["route_custody"] = [_gate(seen, refuse=lambda _c: True)]
        with _mark("kimi-coding"), _config({"provider": "kimi-coding"}), _named():
            assert route_hooks.marked_main_refuses("custom", "k2", task="title", base_url=url)
        assert [(c["provider"], c["model"]) for c in seen] == [("kimi-coding", "k2")]

    def test_with_no_gate_a_kimi_code_client_carries_nothing(self, hooks):
        with _mark("kimi-coding"), _config({"provider": "kimi-coding"}), _named(), \
                patch.object(aux, "resolve_provider_client") as build:
            assert aux._get_cached_client("custom", "k2", base_url=KIMI_CODE, task="title") == (None, None)
        build.assert_not_called()

    def test_a_built_client_at_kimi_code_carries_nothing(self, hooks):
        with _mark("kimi-coding"), _config({"provider": "kimi-coding"}), _named(), \
                patch.object(aux, "resolve_provider_client", return_value=(_client(KIMI_CODE), "k2")):
            assert aux._get_cached_client("custom", "k2", task="title") == (None, None)

    def test_kimi_code_is_not_a_marked_endpoint_for_another_provider(self, hooks):
        with _mark("deepseek"), _config({"provider": "deepseek"}), _named():
            assert not route_hooks.marked_main_refuses("custom", "k2", task="title", base_url=KIMI_CODE)

    @pytest.mark.parametrize("url", ["https://open.bigmodel.cn/api/paas/v4", "https://open.bigmodel.cn/api/coding/paas/v4"])
    def test_a_zai_region_the_probe_can_select_is_a_marked_endpoint(self, hooks, url):
        with _mark("zai"), _config({"provider": "zai"}), _named():
            assert route_hooks.marked_main_refuses("custom", "glm-x", task="title", base_url=url)


class TestEntryEndpoint:
    """F-B: a chain entry is judged by its own URL, its ``providers`` entry's URL and its built client."""

    def test_an_entry_with_a_base_url_at_the_marked_endpoint_needs_an_allow(self):
        mark, config, named = _zai_mark()
        with mark, config, named:
            assert route_hooks.entry_requires_custody({"provider": "custom", "model": "m", "base_url": ZAI})
            assert not route_hooks.entry_requires_custody({"provider": "custom", "model": "m", "base_url": LOCAL})

    def test_an_entry_whose_providers_entry_is_at_the_marked_endpoint_needs_an_allow(self):
        with _mark("zai"), _config({"provider": "zai"}), _named({"relay": {"base_url": ZAI + "/"}}):
            assert route_hooks.entry_requires_custody({"provider": "relay", "model": "m"})
            assert not route_hooks.entry_requires_custody({"provider": "other", "model": "m"})

    def test_a_providers_lookup_that_fails_needs_an_allow(self):
        with _mark("zai"), _config({"provider": "zai"}), _named(fail=("relay",)):
            assert route_hooks.entry_requires_custody({"provider": "relay", "model": "m"})

    def test_with_no_mark_the_url_needs_no_allow(self):
        with _mark("zai", marked=False), _config({"provider": "zai"}), _named():
            assert not route_hooks.entry_requires_custody({"provider": "custom", "model": "m", "base_url": ZAI})

    def test_the_gate_sees_such_an_entry_as_the_marked_route(self, hooks):
        seen = []
        hooks["route_custody"] = [_gate(seen)]
        mark, config, named = _zai_mark()
        with mark, config, named:
            entry = {"provider": "custom", "model": "m", "base_url": ZAI}
            assert not route_hooks.auxiliary_refuses(entry, task="compression", failed_provider="zai")
        assert seen == [{"provider": "zai", "model": "m", "slot_provider": "custom"}]

    @staticmethod
    def _task_chain(chain, built):
        with patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}), \
                patch.object(aux, "_is_provider_unhealthy", return_value=False), \
                patch.object(aux, "_context_too_small", return_value=None), \
                patch.object(aux, "_resolve_fallback_entry", return_value=built) as build:
            return aux._try_configured_fallback_chain("compression", "zai"), build

    @staticmethod
    def _main_chain(chain, built):
        with patch("hermes_cli.config.load_config_readonly", return_value={"fallback_providers": chain}), \
                patch.object(aux, "_is_provider_unhealthy", return_value=False), \
                patch.object(aux, "_context_too_small", return_value=None), \
                patch.object(aux, "_resolve_fallback_entry", return_value=built) as build:
            return aux._try_main_fallback_chain("compression", "zai"), build

    @pytest.mark.parametrize("walk", ["_task_chain", "_main_chain"])
    def test_with_no_gate_an_entry_at_the_marked_endpoint_is_not_built(self, hooks, walk):
        mark, config, named = _zai_mark()
        with mark, config, named:
            result, build = getattr(self, walk)([{"provider": "custom", "model": "m", "base_url": ZAI}],
                                                (_client(ZAI), "m"))
        assert result == (None, None, "")
        build.assert_not_called()

    @pytest.mark.parametrize("walk", ["_task_chain", "_main_chain"])
    def test_an_entry_whose_client_reaches_the_marked_endpoint_carries_nothing(self, hooks, walk):
        mark, config, named = _zai_mark()
        with mark, config, named:
            result, _build = getattr(self, walk)([{"provider": "custom", "model": "m"}], (_client(ZAI), "m"))
        assert result == (None, None, "")

    @pytest.mark.parametrize("walk", ["_task_chain", "_main_chain"])
    def test_an_entry_whose_client_is_elsewhere_is_used(self, hooks, walk):
        local = _client(LOCAL)
        mark, config, named = _zai_mark()
        with mark, config, named:
            (client, model, _label), _build = getattr(self, walk)([{"provider": "custom", "model": "m"}], (local, "m"))
        assert (client, model) == (local, "m")

    def test_an_entry_judged_before_the_build_is_not_asked_again(self, hooks):
        seen, built = [], _client(ZAI)
        hooks["route_custody"] = [_gate(seen)]
        mark, config, named = _zai_mark()
        with mark, config, named:
            (client, _model, _label), _build = self._task_chain([{"provider": "custom", "model": "m", "base_url": ZAI}],
                                                                (built, "m"))
        assert client is built
        assert [(c["provider"], c["model"]) for c in seen] == [("zai", "m")]


class TestVisionBuiltClient:
    """F-C: every vision path judges the client it built, including when the gate saw its model."""

    def test_step_one_is_judged_at_the_runtime_base_url_before_the_build(self, hooks):
        runtime = {"provider": "custom", "model": "v-m", "base_url": ZAI}
        mark, config, named = _zai_mark()
        with mark, config, named, patch.object(aux, "_resolve_provider_vision_default", return_value=None), \
                patch.object(aux, "_vision_main_provider_client", return_value=(None, None)) as build, \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(None, None)):
            assert aux._vision_auto_route(runtime, None, None, False) == (None, None, None)
        build.assert_not_called()

    def test_step_one_judges_the_client_it_built_on_the_model_the_gate_saw(self, hooks):
        runtime = {"provider": "deepseek", "model": "v-m"}
        mark, config, named = _zai_mark()
        with mark, config, named, patch.object(aux, "_resolve_provider_vision_default", return_value=None), \
                patch.object(aux, "_vision_main_provider_client", return_value=(_client(ZAI), "v-m")):
            assert aux._vision_auto_route(runtime, None, None, False)[1:] == (None, None)

    def test_a_strict_backend_is_judged_at_its_client_on_the_model_the_gate_saw(self, hooks):
        mark, config, named = _zai_mark()
        with mark, config, named, \
                patch.object(aux, "_resolve_strict_vision_backend", return_value=(_client(ZAI), "v-m")):
            assert aux.resolve_vision_provider_client("openrouter", "v-m")[1:] == (None, None)

    def test_a_strict_backend_elsewhere_is_sent(self, hooks):
        local = _client("https://openrouter.ai/api/v1")
        mark, config, named = _zai_mark()
        with mark, config, named, patch.object(aux, "_resolve_strict_vision_backend", return_value=(local, "v-m")):
            assert aux.resolve_vision_provider_client("openrouter", "v-m")[1:] == (local, "v-m")

    def test_step_one_judged_at_its_url_is_not_asked_again(self, hooks):
        seen, built = [], _client(ZAI)
        hooks["route_custody"] = [_gate(seen)]
        runtime = {"provider": "custom", "model": "v-m", "base_url": ZAI}
        mark, config, named = _zai_mark()
        with mark, config, named, patch.object(aux, "_resolve_provider_vision_default", return_value=None), \
                patch.object(aux, "_vision_main_provider_client", return_value=(built, "v-m")):
            assert aux._vision_auto_route(runtime, None, None, False) == ("custom", built, "v-m")
        assert [(c["provider"], c["model"]) for c in seen] == [("zai", "v-m")]


class TestAutoClientCache:
    """F-D: a cached ``auto`` client is built again while a chain it walks holds a required entry."""

    @staticmethod
    def _twice(chain):
        built = _client()
        with _mark("", marked=False), patch.object(aux, "_read_main_provider", return_value="deepseek"), \
                patch("hermes_cli.config.load_config_readonly", return_value={"fallback_providers": chain}), \
                patch.object(aux, "resolve_provider_client", return_value=(built, "m")) as build:
            for _ in range(2):
                assert aux._get_cached_client("auto", "m", task="compression") == (built, "m")
        return build.call_count

    def test_a_required_main_chain_entry_rebuilds_on_each_call(self, hooks):
        assert self._twice([{"provider": "hosted", "model": "h", "route_custody": "required"}]) == 2

    def test_a_required_task_chain_entry_rebuilds_on_each_call(self, hooks):
        chain = [{"provider": "hosted", "model": "h", "route_custody": "required"}]
        with patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}):
            assert self._twice([]) == 2

    def test_an_unkeyed_chain_reuses_the_cached_client(self, hooks):
        assert self._twice([{"provider": "hosted", "model": "h"}]) == 1

    def test_an_unreadable_chain_rebuilds_on_each_call(self, hooks):
        with patch("hermes_cli.fallback_config.get_fallback_chain", side_effect=OSError("unreadable")):
            assert self._twice([]) == 2

    def test_the_walk_asks_the_gate_on_each_call(self, hooks):
        seen = []
        hooks["route_custody"] = [_gate(seen)]
        chain = [{"provider": "hosted", "model": "h", "route_custody": "required"}]
        with _mark("", marked=False), patch.object(aux, "_read_main_provider", return_value="deepseek"), \
                patch("hermes_cli.config.load_config_readonly", return_value={"fallback_providers": chain}), \
                patch.object(aux, "_get_auxiliary_task_config", return_value={}), \
                patch.object(aux, "_try_main_provider_route", return_value=None), \
                patch.object(aux, "_warn_stale_openai_base_url"), \
                patch.object(aux, "_is_provider_unhealthy", return_value=False), \
                patch.object(aux, "_context_too_small", return_value=None), \
                patch.object(aux, "_resolve_fallback_entry", return_value=(_client(), "h")):
            for _ in range(2):
                assert aux._get_cached_client("auto", None, task="compression")[0] is not None
        assert [(c["provider"], c["model"]) for c in seen] == [("hosted", "h"), ("hosted", "h")]


class TestClientBaseUrl:
    """F-E: the URL a built client sends to, read from an ``httpx.URL`` and from a wrapped client."""

    def test_an_httpx_url_is_read(self):
        assert route_hooks.client_base_url(SimpleNamespace(base_url=httpx.URL(ZAI))) == ZAI

    @pytest.mark.parametrize("attr", ["_client", "client"])
    def test_a_wrapped_clients_url_is_read(self, attr):
        assert route_hooks.client_base_url(SimpleNamespace(**{attr: SimpleNamespace(base_url=ZAI)})) == ZAI

    def test_a_client_with_an_httpx_url_at_the_marked_endpoint_carries_nothing(self, hooks):
        built = SimpleNamespace(base_url=httpx.URL(ZAI))
        mark, config, named = _zai_mark()
        with mark, config, named:
            assert route_hooks.judged_built_client("custom", built, "m", task="title") == (None, None)

    @pytest.mark.parametrize("attr", ["_client", "client"])
    def test_a_wrapped_client_at_the_marked_endpoint_carries_nothing(self, hooks, attr):
        built = SimpleNamespace(**{attr: SimpleNamespace(base_url=ZAI)})
        mark, config, named = _zai_mark()
        with mark, config, named:
            assert route_hooks.judged_built_client("custom", built, "m", task="title") == (None, None)

    def test_an_auto_slots_own_model_is_judged_at_its_clients_endpoint(self, hooks):
        seen = []
        hooks["route_custody"] = [_gate(seen, refuse=lambda c: c["model"] == "slot-m")]
        mark, config, named = _zai_mark()
        with mark, config, named, \
                patch.object(aux, "_main_route_target", return_value=("custom", "main-m", "", "", "")), \
                patch.object(aux, "_try_main_provider_route", return_value=(_client(ZAI), "main-m", "custom")), \
                patch.object(aux, "_warn_stale_openai_base_url"):
            assert aux.resolve_provider_client("auto", model="slot-m", task="compression") == (None, None)
        assert [(c["provider"], c["model"]) for c in seen] == [("zai", "slot-m")]


class TestSlotProvider:
    """Item 1: ``candidate["provider"]`` stays the marked name, and ``slot_provider`` names the slot."""

    def test_a_slot_at_the_marked_endpoint_names_itself(self, hooks):
        seen = []
        hooks["route_custody"] = [_gate(seen)]
        mark, config, named = _zai_mark()
        with mark, config, named:
            assert not route_hooks.marked_main_refuses("Custom", "m", task="title", base_url=ZAI)
        assert seen == [{"provider": "zai", "model": "m", "slot_provider": "custom"}]

    def test_a_built_client_names_its_slot(self, hooks):
        seen = []
        hooks["route_custody"] = [_gate(seen)]
        mark, config, named = _zai_mark()
        with mark, config, named:
            route_hooks.judged_built_client("deepseek", _client(ZAI), "m", task="title")
        assert seen == [{"provider": "zai", "model": "m", "slot_provider": "deepseek"}]

    def test_an_unmarked_entry_carries_no_slot_provider(self, hooks):
        seen = []
        hooks["route_custody"] = [_gate(seen)]
        with _mark("zai", marked=False):
            route_hooks.auxiliary_refuses({"provider": "hosted", "model": "h"}, task="title", failed_provider="zai")
        assert seen == [{"provider": "hosted", "model": "h"}]
