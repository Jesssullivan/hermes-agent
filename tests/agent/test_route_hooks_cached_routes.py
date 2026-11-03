"""F12 on cached and recorded routes at the marked endpoint (agent/route_hooks.py).

A cached ``auto`` client is judged on each call as its walk judged it, and a refused one is walked
again. A resolution-time record keeps the URLs its entry was configured with and resolved to. The
main agent's own route at the marked endpoint under another name is judged as the marked route, and so
is a kept fallback at the URL the agent sends to. The last-resort hop to the main model from a kept
fallback is judged at its own provider and URL. A CLI shell with no session id still binds a side
route. The Copilot API hosts the token exchange names are part of a ``copilot`` mark. A ``vision`` slot with a
``base_url`` is judged on the URL its built client sends to. Each probe drives the production code
with stub clients and a stub gate. No provider is called."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux
from agent import route_hooks
from hermes_cli.plugins import PluginManager

ZAI = "https://api.z.ai/api/paas/v4"
KIMI_CODE = "https://api.kimi.com/coding"
LOCAL = "http://127.0.0.1:8080/v1"
COPILOT_INDIVIDUAL = "https://api.individual.githubcopilot.com"


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
    for name in ("KIMI_BASE_URL", "GLM_BASE_URL", "COPILOT_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with route_hooks._SESSIONS_LOCK:
        saved = dict(route_hooks._SESSIONS)
        route_hooks._SESSIONS.clear()
    yield
    with route_hooks._SESSIONS_LOCK:
        route_hooks._SESSIONS.clear()
        route_hooks._SESSIONS.update(saved)
    route_hooks._ROUTE_SCOPE.reset(token)


def _marked(provider, marked=True):
    """``model.route_custody`` on *provider*, read without a config file."""
    return [patch("agent.route_hooks._main_route_marker", return_value=(provider, marked)),
            patch("hermes_cli.config.read_raw_config_readonly", return_value={"model": {"provider": provider}}),
            patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=None)]


class _Gate:
    """Records each candidate; refuses while ``refuse`` is set, or answers ``answer``."""

    def __init__(self, answer="allow"):
        self.seen, self.refuse, self.answer = [], False, answer

    def __call__(self, candidate, **_kw):
        self.seen.append(dict(candidate))
        return {"action": "skip"} if self.refuse else (self.answer and {"action": self.answer})


def _client(base_url=None):
    client = MagicMock()
    if base_url is not None:
        client.base_url = base_url
    return client


class _Walk:
    """``_get_cached_client("auto")`` through the real walk: the main route *main* (``None`` for none),
    the main fallback chain, which a test may change between calls, and stub builds per entry."""

    def __init__(self, *, main=None, chain=(), built=None, main_target=("deepseek", "main-m", "", "", "")):
        self.main, self.chain, self.built, self.main_target = main, list(chain), dict(built or {}), main_target
        self.builds = []

    def _build(self, entry):
        self.builds.append(entry["provider"])
        return self.built[entry["provider"]], entry.get("model")

    def call(self, *marks, model=None, provider="auto"):
        patches = [*marks,
                   patch.object(aux, "_main_route_target", return_value=self.main_target),
                   patch.object(aux, "_try_main_provider_route", return_value=self.main),
                   patch.object(aux, "_read_main_provider", return_value=self.main_target[0]),
                   patch.object(aux, "_warn_stale_openai_base_url"),
                   patch.object(aux, "_get_auxiliary_task_config", return_value={}),
                   patch("hermes_cli.config.load_config_readonly", return_value={"fallback_providers": self.chain}),
                   patch.object(aux, "_is_provider_unhealthy", return_value=False),
                   patch.object(aux, "_context_too_small", return_value=None),
                   patch.object(aux, "_resolve_fallback_entry", side_effect=self._build)]
        for item in patches:
            item.start()
        try:
            return aux._get_cached_client(provider, model, task="compression")
        finally:
            for item in reversed(patches):
                item.stop()


class TestCachedAutoClient:
    """R1: each cache hit of an ``auto`` client is judged; a refused hit is dropped and walked again."""

    def test_a_main_route_client_at_the_marked_endpoint_is_judged_on_each_hit(self, hooks):
        gate, built = _Gate(), _client(KIMI_CODE)
        hooks["route_custody"] = [gate]
        walk = _Walk(main=(built, "k2", "custom"), main_target=("custom", "k2", KIMI_CODE, "", ""))
        assert walk.call(*_marked("kimi-coding"))[0] is built
        gate.refuse = True
        assert walk.call(*_marked("kimi-coding")) == (None, None), "the cached client was served"
        assert {(c["provider"], c.get("slot_provider")) for c in gate.seen} == {("kimi-coding", "custom")}

    @pytest.mark.parametrize("spelling", ["Auto", "AUTO", " auto "])
    def test_a_hit_of_any_spelling_of_auto_is_judged(self, hooks, spelling):
        gate, built = _Gate(), _client(KIMI_CODE)
        hooks["route_custody"] = [gate]
        walk = _Walk(main=(built, "k2", "custom"), main_target=("custom", "k2", KIMI_CODE, "", ""))
        assert walk.call(*_marked("kimi-coding"), provider=spelling)[0] is built
        gate.refuse = True
        assert walk.call(*_marked("kimi-coding"), provider=spelling) == (None, None), "the cached client was served"

    def test_an_admitted_hit_is_served_from_the_cache(self, hooks):
        gate, built = _Gate(), _client(KIMI_CODE)
        hooks["route_custody"] = [gate]
        walk = _Walk(main=(built, "k2", "custom"), main_target=("custom", "k2", KIMI_CODE, "", ""))
        assert walk.call(*_marked("kimi-coding"))[0] is built
        assert walk.call(*_marked("kimi-coding"))[0] is built
        assert len(gate.seen) == 2, "the walk and the hit each ask once"
        assert len(aux._client_cache) == 1

    def test_a_chain_entry_client_at_the_marked_endpoint_is_judged_on_each_hit(self, hooks):
        gate, built = _Gate(), _client(ZAI)
        hooks["route_custody"] = [gate]
        walk = _Walk(chain=[{"provider": "custom", "model": "m"}], built={"custom": built})
        assert walk.call(*_marked("zai"))[0] is built
        gate.refuse = True
        assert walk.call(*_marked("zai")) == (None, None), "the cached client was served"
        assert ("zai", "custom") in {(c["provider"], c.get("slot_provider")) for c in gate.seen}

    def test_a_hit_is_judged_on_the_slots_own_model(self, hooks):
        gate, built = _Gate(), _client(ZAI)
        hooks["route_custody"] = [gate]
        walk = _Walk(chain=[{"provider": "zai", "model": "m"}], built={"zai": built})
        assert walk.call(*_marked("zai"), model="slot-m")[0] is built
        hooks["route_custody"] = [lambda candidate, **_kw: {"action": "skip" if candidate["model"] == "slot-m"
                                                            else "allow"}]
        assert walk.call(*_marked("zai"), model="slot-m")[0] is None, "the slot's own model was not judged"

    def test_a_mark_added_after_the_cache_reaches_the_hit(self, hooks):
        built = _client(KIMI_CODE)
        walk = _Walk(main=(built, "k2", "custom"), main_target=("custom", "k2", KIMI_CODE, "", ""))
        assert walk.call(*_marked("", marked=False))[0] is built
        assert walk.call(*_marked("kimi-coding")) == (None, None), "a marked endpoint carried a call with no gate"

    def test_a_loaded_gate_is_asked_on_each_hit_of_an_unkeyed_entry(self, hooks):
        gate, built = _Gate(), _client(LOCAL)
        hooks["route_custody"] = [gate]
        walk = _Walk(chain=[{"provider": "hosted", "model": "h"}], built={"hosted": built})
        assert walk.call(*_marked("", marked=False))[0] is built
        gate.refuse = True
        assert walk.call(*_marked("", marked=False)) == (None, None), "the cached client was served"
        assert walk.builds == ["hosted"], "a refused entry was built again"
        assert [c["provider"] for c in gate.seen] == ["hosted", "hosted", "hosted"]

    def test_a_key_added_to_the_entry_after_the_cache_reaches_the_hit(self, hooks):
        built = _client(LOCAL)
        walk = _Walk(chain=[{"provider": "hosted", "model": "h"}], built={"hosted": built})
        assert walk.call(*_marked("", marked=False))[0] is built
        walk.chain = [{"provider": "hosted", "model": "h", "route_custody": "required"}]
        assert walk.call(*_marked("", marked=False)) == (None, None), "a required entry carried a call with no gate"

    def test_an_unreadable_chain_judges_the_hit_as_required(self, hooks):
        built = _client(LOCAL)
        walk = _Walk(chain=[{"provider": "hosted", "model": "h"}], built={"hosted": built})
        assert walk.call(*_marked("", marked=False))[0] is built
        unreadable = patch("hermes_cli.fallback_config.get_fallback_chain", side_effect=OSError("unreadable"))
        assert walk.call(unreadable, *_marked("", marked=False)) == (None, None)

    def test_a_refused_hit_walks_on_to_the_next_admitted_entry(self, hooks):
        gate, first, second = _Gate(), _client(LOCAL), _client("http://127.0.0.1:9090/v1")
        hooks["route_custody"] = [gate]
        walk = _Walk(chain=[{"provider": "hosted", "model": "h"}, {"provider": "spare", "model": "s"}],
                     built={"hosted": first, "spare": second})
        assert walk.call(*_marked("", marked=False))[0] is first
        hooks["route_custody"] = [lambda candidate, **_kw: {"action": "skip" if candidate["provider"] == "hosted"
                                                            else "allow"}]
        assert walk.call(*_marked("", marked=False))[0] is second
        assert walk.call(*_marked("", marked=False))[0] is second, "the next call reuses the new client"

    def test_a_hit_whose_entry_left_the_chain_walks_again(self, hooks):
        old, new = _client(LOCAL), _client("http://127.0.0.1:9090/v1")
        walk = _Walk(chain=[{"provider": "hosted", "model": "h"}], built={"hosted": old, "spare": new})
        assert walk.call(*_marked("", marked=False))[0] is old
        walk.chain = [{"provider": "spare", "model": "s"}]
        assert walk.call(*_marked("", marked=False))[0] is new

    def test_an_async_wrapper_keeps_the_entry_it_was_built_on(self):
        built, wrapper = SimpleNamespace(), SimpleNamespace()
        with _marked("", marked=False)[0]:
            route_hooks.judged_entry_client({"provider": "hosted", "model": "h", "api_key": "secret"}, built, "h",
                                            task="compression")
        assert route_hooks.carry_auto_entry(built, wrapper, "h") == (wrapper, "h")
        assert wrapper._hermes_route_entry == {"provider": "hosted", "model": "h"}, "a secret was stamped"

    def test_the_auto_branch_stamps_the_client_it_returns(self, hooks):
        built, wrapper = _client(LOCAL), SimpleNamespace(base_url=LOCAL)
        chain = [{"provider": "hosted", "model": "h", "route_custody": "required"}]
        walk = _Walk(chain=chain, built={"hosted": built})
        hooks["route_custody"] = [_Gate()]
        with patch.object(aux, "_route_client", return_value=(wrapper, "h")):
            assert walk.call(*_marked("", marked=False))[0] is wrapper
        assert wrapper._hermes_route_entry["route_custody"] == "required"


class TestRecordedEndpoint:
    """R2: a resolution-time record keeps its URLs, so later checks judge it at the marked endpoint."""

    @staticmethod
    def _agent(record, **attrs):
        agent = SimpleNamespace(provider="custom", model="m", session_id="s-1", platform="", base_url="",
                                _provider_fallback_active=False, _route_resolution_entry=None, **attrs)
        route_hooks.bind_resolution_entry(agent, record)
        return agent

    def test_the_record_keeps_the_urls_with_no_secret(self):
        entry = {"provider": "custom", "model": "m", "base_url": "https://u:pw@api.z.ai/api/paas/v4?key=k#f"}
        record = route_hooks.resolution_record(entry, {"provider": "custom", "base_url": ZAI + "?token=t"})
        assert record["base_url"] == ZAI and record["resolved_base_url"] == ZAI

    @pytest.mark.parametrize("entry,runtime", [
        ({"provider": "custom", "model": "m", "base_url": ZAI}, {"provider": "custom"}),
        ({"provider": "custom", "model": "m"}, {"provider": "custom", "base_url": ZAI}),
    ])
    def test_a_later_turn_judges_the_record_as_the_marked_route(self, hooks, entry, runtime):
        gate = _Gate(answer=None)
        hooks["route_custody"] = [gate]
        marks = _marked("zai")
        for item in marks:
            item.start()
        try:
            agent = self._agent(route_hooks.resolution_record(entry, runtime))
            with pytest.raises(route_hooks.RouteCustodyRefused):
                route_hooks.check_out_of_turn_route(agent, phase="compaction")
        finally:
            for item in reversed(marks):
                item.stop()
        assert gate.seen == [{"provider": "zai", "model": "m", "slot_provider": "custom"}]

    def test_a_side_agent_on_a_kept_fallback_carries_the_url(self):
        parent = SimpleNamespace(provider="custom", model="m", base_url=ZAI, _provider_fallback_active=True,
                                 _route_fallback_entry={"provider": "custom", "model": "m"})
        record = route_hooks.side_route_record(parent)
        assert record["resolved_base_url"] == ZAI
        marks = _marked("zai")
        for item in marks:
            item.start()
        try:
            assert route_hooks.entry_requires_custody(record)
        finally:
            for item in reversed(marks):
                item.stop()

    def test_a_delegated_child_judges_the_record_at_the_marked_endpoint(self, hooks):
        record = route_hooks.resolution_record({"provider": "custom", "model": "m", "base_url": ZAI}, {"provider": "custom"})
        child = SimpleNamespace(provider="custom", model="m", session_id="child", platform="", base_url="",
                                _provider_fallback_active=False, _route_resolution_entry=None)
        marks = _marked("zai")
        for item in marks:
            item.start()
        try:
            route_hooks.inherit_child_route_custody(child, record, SimpleNamespace(session_id="parent"))
            with pytest.raises(route_hooks.RouteCustodyRefused):
                route_hooks.check_out_of_turn_route(child, phase="compaction")
        finally:
            for item in reversed(marks):
                item.stop()

    def test_the_init_walker_records_the_url_its_client_resolved_to(self, hooks):
        from agent.agent_init import _routed_client_kwargs

        def resolve(provider, model=None, **_kw):
            return (None, None) if provider == "anthropic" else (_client(ZAI), model)

        agent = SimpleNamespace(provider="anthropic", requested_provider="anthropic", model="opus",
                                api_mode="chat_completions", _fallback_activated=False)
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve):
            _routed_client_kwargs(agent, [{"provider": "custom", "model": "m"}], None)
        record = agent._route_resolution_entry
        assert record["resolved_base_url"] == ZAI and "base_url" not in record
        marks = _marked("zai")
        for item in marks:
            item.start()
        try:
            assert route_hooks.entry_requires_custody(record), "the init record was not judged at the marked endpoint"
        finally:
            for item in reversed(marks):
                item.stop()

    def test_a_record_elsewhere_needs_no_allow(self):
        record = route_hooks.resolution_record({"provider": "custom", "model": "m", "base_url": LOCAL}, {"provider": "custom"})
        marks = _marked("zai")
        for item in marks:
            item.start()
        try:
            assert not route_hooks.entry_requires_custody(record)
        finally:
            for item in reversed(marks):
                item.stop()


class TestMainRouteAtTheMarkedEndpoint:
    """R5: the main agent's route under another name at the marked endpoint is the marked route."""

    @staticmethod
    def _agent(base_url):
        return SimpleNamespace(provider="custom", model="k2", base_url=base_url, session_id="s-2", platform="",
                               _provider_fallback_active=False, _route_resolution_entry=None, _fallback_chain=[])

    def _run(self, fn, *marks):
        for item in marks:
            item.start()
        try:
            return fn()
        finally:
            for item in reversed(marks):
                item.stop()

    def test_with_no_gate_the_route_sends_nothing(self, hooks):
        agent = self._agent(KIMI_CODE + "/v1")
        assert self._run(lambda: route_hooks.agent_custody_applies(agent), *_marked("kimi-coding"))
        with pytest.raises(route_hooks.RouteCustodyRefused):
            self._run(lambda: route_hooks.check_request_route(agent, [{"role": "user", "content": "q"}]),
                      *_marked("kimi-coding"))

    def test_the_gate_sees_the_marked_route(self, hooks):
        gate = _Gate()
        hooks["route_custody"] = [gate]
        agent = self._agent(KIMI_CODE)
        assert not self._run(lambda: route_hooks.check_request_route(agent, []), *_marked("kimi-coding"))
        assert gate.seen == [{"provider": "kimi-coding", "model": "k2", "slot_provider": "custom"}]

    def test_a_route_elsewhere_is_not_the_marked_route(self, hooks):
        agent = self._agent(LOCAL)
        assert not self._run(lambda: route_hooks.agent_custody_applies(agent), *_marked("kimi-coding"))
        assert self._run(lambda: route_hooks._active_entry(agent), *_marked("kimi-coding")) is None


class TestKeptFallbackAtTheMarkedEndpoint:
    """R5: a kept fallback whose entry names no URL is judged at the URL the agent sends to."""

    @staticmethod
    def _agent(base_url):
        return SimpleNamespace(provider="custom", model="m", base_url=base_url, session_id="s-3", platform="",
                               _provider_fallback_active=True, _route_fallback_entry={"provider": "custom", "model": "m"},
                               _route_resolution_entry=None, _fallback_chain=[], _rate_limited_until=0,
                               _restore_primary_runtime=lambda: False)

    def test_with_no_gate_the_fallback_sends_nothing(self, hooks):
        agent = self._agent(ZAI)
        with pytest.raises(route_hooks.RouteCustodyRefused):
            TestMainRouteAtTheMarkedEndpoint()._run(
                lambda: route_hooks.check_request_route(agent, [{"role": "user", "content": "q"}]), *_marked("zai"))

    def test_the_gate_sees_the_marked_route_as_required(self, hooks):
        seen = []
        hooks["route_custody"] = [lambda candidate, required, **_kw: seen.append((dict(candidate), required))
                                  or {"action": "allow"}]
        agent = self._agent(ZAI)
        assert not TestMainRouteAtTheMarkedEndpoint()._run(
            lambda: route_hooks.check_request_route(agent, []), *_marked("zai"))
        assert seen == [({"provider": "zai", "model": "m", "slot_provider": "custom"}, True)]
        assert agent._route_fallback_entry == {"provider": "custom", "model": "m"}, "the kept entry was changed"

    def test_a_fallback_elsewhere_is_not_the_marked_route(self, hooks):
        agent = self._agent(LOCAL)
        assert TestMainRouteAtTheMarkedEndpoint()._run(lambda: route_hooks._active_entry(agent), *_marked("zai")) is None


class TestMainHopFromAKeptFallback:
    """The last-resort hop to the main model is judged at its own provider and URL. Only the custody
    key comes from the kept fallback the scope agent runs on, not its URLs."""

    @staticmethod
    def _hop(provider, base_url=""):
        client = _client()
        return client, [*_marked("zai"), patch.multiple(
            aux, _read_main_provider=lambda: provider, _read_main_model=lambda: "or-m",
            _custom_health_base_url=lambda *_a: base_url, _is_provider_unhealthy=lambda *_a: False,
            main_model_fallback_refused=lambda *_a: False), patch.object(aux, "resolve_provider_client",
                                                                         return_value=(client, "or-m"))]

    def _send(self, agent, provider, base_url=""):
        route_hooks._bind_scope(agent)
        client, marks = self._hop(provider, base_url)
        sent = TestMainRouteAtTheMarkedEndpoint()._run(
            lambda: aux._try_main_agent_model_fallback("zai", "compression", failed_model="glm"), *marks)
        return client, sent

    def test_a_hop_to_an_unmarked_provider_is_not_refused_with_no_gate(self, hooks):
        client, sent = self._send(TestKeptFallbackAtTheMarkedEndpoint._agent(ZAI), "openrouter")
        assert sent[0] is client

    def test_the_gate_sees_the_hops_own_provider(self, hooks):
        seen = []
        hooks["route_custody"] = [lambda candidate, required, **_kw: seen.append((dict(candidate), required))
                                  or {"action": "allow"}]
        self._send(TestKeptFallbackAtTheMarkedEndpoint._agent(ZAI), "openrouter")
        assert seen == [({"provider": "openrouter", "model": "or-m"}, False)]

    def test_a_hop_whose_own_url_is_the_marked_endpoint_is_judged_as_marked(self, hooks):
        assert self._send(TestKeptFallbackAtTheMarkedEndpoint._agent(LOCAL), "custom", ZAI)[1] == (None, None, "")
        seen = []
        hooks["route_custody"] = [lambda candidate, required, **_kw: seen.append((dict(candidate), required))
                                  or {"action": "allow"}]
        self._send(TestKeptFallbackAtTheMarkedEndpoint._agent(LOCAL), "custom", ZAI)
        assert seen == [({"provider": "zai", "model": "or-m", "slot_provider": "custom"}, True)]


class TestSideRouteOnAShellWithNoSession:
    """ce9062ce: a CLI shell with no session id still gives a side call its route context."""

    @staticmethod
    def _shell(**attrs):
        from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
        shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
        shell.__dict__.update(attrs)
        return shell

    def test_side_route_with_no_session_id(self):
        assert self._shell()._side_route() == {"record": None, "session_id": None, "platform": "cli"}

    def test_side_route_carries_the_session_and_record(self):
        record = {"provider": "custom", "model": "m"}
        shell = self._shell(session_id="s-4", _route_entry_record=record)
        assert shell._side_route() == {"record": record, "session_id": "s-4", "platform": "cli"}

    def test_bind_side_route_with_no_session_id(self):
        with patch("agent.route_hooks.inherit_route_custody") as inherit:
            self._shell()._bind_side_route(agent := SimpleNamespace())
        inherit.assert_called_once_with(agent, None, session_id=None)


class TestCopilotEndpoints:
    """R3: a ``copilot`` mark covers the API hosts the token exchange names."""

    def test_a_plan_host_is_a_marked_endpoint(self, hooks):
        marks = _marked("copilot")
        assert TestMainRouteAtTheMarkedEndpoint()._run(
            lambda: route_hooks.marked_main_refuses("custom", "gpt-4o", task="title", base_url=COPILOT_INDIVIDUAL), *marks)

    def test_a_host_an_exchange_returned_is_a_marked_endpoint(self, hooks, monkeypatch):
        from hermes_cli import copilot_auth
        monkeypatch.setattr(copilot_auth, "_jwt_cache", {"fp": ("api-token", 0.0, "https://api.proxy.example.test")})
        assert copilot_auth.exchanged_api_urls() == ["https://api.proxy.example.test"]
        assert TestMainRouteAtTheMarkedEndpoint()._run(
            lambda: route_hooks.marked_main_refuses("custom", "gpt-4o", task="title",
                                                    base_url="https://api.proxy.example.test/v1"), *_marked("copilot"))

    def test_a_plan_host_is_not_marked_for_another_provider(self, hooks):
        assert not TestMainRouteAtTheMarkedEndpoint()._run(
            lambda: route_hooks.marked_main_refuses("custom", "gpt-4o", task="title", base_url=COPILOT_INDIVIDUAL),
            *_marked("zai"))


class TestVisionBaseUrlBranch:
    """R4 and survivor M: a vision slot with a ``base_url`` is judged on its built client."""

    def _resolve(self, built):
        marks = [*_marked("zai"),
                 patch.object(aux, "_resolve_task_provider_model", return_value=("nous", "v-m", LOCAL, "k", None)),
                 patch.object(aux, "resolve_provider_client", return_value=(built, "v-m"))]
        return TestMainRouteAtTheMarkedEndpoint()._run(lambda: aux.resolve_vision_provider_client(), *marks)

    def test_a_backend_that_ignores_the_base_url_and_reaches_the_mark_sends_nothing(self, hooks):
        assert self._resolve(_client(ZAI))[1:] == (None, None)

    def test_a_backend_at_the_configured_url_is_sent(self, hooks):
        built = _client(LOCAL)
        assert self._resolve(built)[1:] == (built, "v-m")

    def test_no_client_sends_nothing(self):
        assert route_hooks.vision_send_refused("openrouter", None, "m", "m") is True
        assert aux._finalize_vision_client("openrouter", None, "default-m", None, False) == ("openrouter", None, None)


def test_hooks_doc_lists_moa_advisors_as_not_covered():
    """R6: the docs name what a ``moa`` mark leaves out."""
    doc = Path(__file__).resolve().parents[2] / "website/docs/user-guide/features/hooks.md"
    not_covered = doc.read_text(encoding="utf-8").split("**Not covered:**", 1)[1]
    assert "**MoA advisors under a `moa` mark.**" in not_covered
