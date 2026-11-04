"""F12 on the client a route builds, and on the URLs a route cannot parse (agent/route_hooks.py).

The ``auto`` walk's main step is judged on the provider it declares and the runtime ``base_url`` before
the build. A ``custom`` main with no endpoint of its own falls through to the first API-key provider
that has a key, which can be the marked endpoint, so the built client is judged at the URL it sends to,
as the last-resort hop's is. The hop judges that client on the model it resolved. A route the mark
names is judged again when its client sends another model than the one judged before the build. A URL that
``urlsplit`` rejects names no endpoint: with no gate and no mark nothing applies, and under either the
route needs an allow. Vision enrichment, a side question and ``session.mutate compress`` each judge
the runtime at its ``base_url``. Each probe drives the production code with stub clients and a stub
gate. No provider is called."""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Imported before any test patches agent.context_compressor.ContextCompressor. agent_init binds that name
# at import, so a first import under the patch would keep the stub for every later test in the process.
import agent.agent_init
from agent import auxiliary_client as aux
from agent import route_hooks
from hermes_cli.plugins import PluginManager
from tests.agent.test_route_hooks_cached_routes import LOCAL, ZAI, _client, _marked

TASKS = ("compression", "title_generation", "side_question")


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


def _run(fn, *marks):
    for item in marks:
        item.start()
    try:
        return fn()
    finally:
        for item in reversed(marks):
            item.stop()


def _gate(seen, refuse=()):
    def custody(candidate, required, **_kw):
        seen.append((dict(candidate), required))
        return {"action": "skip" if candidate["provider"] in refuse else "allow"}
    return [custody]


def _custom_main(built, custom_url=None):
    """A ``custom`` main runtime with no ``base_url``. ``resolve_provider_client`` itself runs: it builds
    on the custom endpoint config or env names (*custom_url*), and with none it falls through to the
    first API-key provider, which builds *built*."""
    endpoint = {} if custom_url else {"_try_custom_endpoint": lambda: (None, None)}
    return [patch.multiple(
        aux, _main_route_target=lambda _runtime, _task: ("custom", "m", "", "", ""),
        _resolve_custom_runtime=lambda: (custom_url, "sk-fixture" if custom_url else None, None),
        _resolve_api_key_provider=lambda: (built, "m"), _is_provider_unhealthy=lambda *_a, **_k: False,
        _warn_stale_openai_base_url=lambda *_a, **_k: None, _get_auxiliary_task_config=lambda *_a, **_k: {},
        _read_main_provider=lambda: "custom", **endpoint),
        patch("hermes_cli.config.load_config_readonly", return_value={})]


class TestAutoWalkMainStep:
    """The main step of an ``auto`` walk is judged at the URL its built client sends to."""

    @pytest.mark.parametrize("task", TASKS)
    def test_a_client_at_the_marked_endpoint_needs_an_allow(self, hooks, task):
        marks = [*_marked("zai"), *_custom_main(_client(ZAI))]
        assert _run(lambda: aux._get_cached_client("auto", None, task=task), *marks) == (None, None)
        assert _run(lambda: aux._get_cached_client("auto", None, task=task), *marks) == (None, None), "a walk again"

    @pytest.mark.parametrize("task", TASKS)
    def test_the_gate_sees_the_marked_route(self, hooks, task):
        seen, built = [], _client(ZAI)
        hooks["route_custody"] = _gate(seen)
        assert _run(lambda: aux._get_cached_client("auto", None, task=task), *_marked("zai"), *_custom_main(built))[0] is built
        assert seen[-1] == ({"provider": "zai", "model": "m", "slot_provider": "custom"}, True)

    def test_the_gate_sees_the_model_the_main_step_resolved(self, hooks):
        seen, built = [], _client(ZAI)
        hooks["route_custody"] = _gate(seen)
        marks = [*_marked("zai"), *_custom_main(built),
                 patch.object(aux, "_try_main_provider_route", return_value=(built, "glm-resolved", "custom"))]
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)[0] is built
        assert seen == [({"provider": "zai", "model": "glm-resolved", "slot_provider": "custom"}, True)]

    def test_a_refusal_drops_the_client_and_ends_the_walk(self, hooks):
        hooks["route_custody"] = _gate([], refuse={"zai"})
        marks = [*_marked("zai"), *_custom_main(_client(ZAI))]
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks) == (None, None)

    def test_a_custom_endpoint_from_config_at_the_mark_is_judged(self, hooks):
        """The endpoint comes from config or env, not the runtime, so only the built client names it."""
        marks = [*_marked("zai"), *_custom_main(_client(LOCAL), custom_url=ZAI)]
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks) == (None, None)
        seen = []
        hooks["route_custody"] = _gate(seen)
        got = _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)
        assert route_hooks.client_base_url(got[0]).rstrip("/") == ZAI
        assert seen[-1] == ({"provider": "zai", "model": "m", "slot_provider": "custom"}, True)

    def test_a_client_elsewhere_is_served_with_no_gate(self, hooks):
        """A control: the judgement reads the built client's URL, and a URL off the mark is not refused."""
        built = _client(LOCAL)
        marks = [*_marked("zai"), *_custom_main(built)]
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)[0] is built
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)[0] is built

    def test_with_no_mark_the_client_is_served_as_before(self, hooks):
        built = _client(ZAI)
        marks = [*_marked("zai", False), *_custom_main(built)]
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)[0] is built


def _side_question_oneshot():
    from agent.side_question import answer_side_question
    with patch("agent.side_question._side_question_task_config", return_value={}):
        return answer_side_question("q", [{"role": "user", "content": "hi"}], parent_agent=None,
                                    main_runtime={"provider": "custom", "model": "m"},
                                    route={"record": None, "session_id": "s-oneshot"})


def _title():
    from agent.title_generator import generate_title
    with patch("agent.title_generator._auto_title_enabled", return_value=True):
        return generate_title("how do I sort a list in place")


def _compression():
    return aux.call_llm(task="compression", messages=[{"role": "user", "content": "summarize this"}])


class TestAutoWalkEntryPoints:
    """Compression, titles and the side question's one-shot each reach the ``auto`` walk through
    ``call_llm``. With the main step on the marked endpoint and no gate, nothing is sent."""

    @pytest.mark.parametrize("entry", [_compression, _title, _side_question_oneshot],
                             ids=["compression", "title", "side-question-oneshot"])
    def test_nothing_is_sent_to_the_marked_endpoint(self, hooks, entry):
        built = _client(ZAI)
        with contextlib.suppress(Exception):
            _run(entry, *_marked("zai"), *_custom_main(built))
        built.chat.completions.create.assert_not_called()


class TestMainHopResolvedModel:
    """The hop judges its built client on the model that client sends, not the one it declared."""

    def test_the_gate_sees_the_resolved_model_at_the_built_url(self, hooks):
        seen, client = [], _client(ZAI)
        hooks["route_custody"] = _gate(seen)
        route_hooks._bind_scope(SimpleNamespace(provider="custom", model="m", base_url=LOCAL, session_id="s-h",
                                                platform="", _provider_fallback_active=False,
                                                _route_resolution_entry=None, _fallback_chain=[]))
        marks = [*_marked("zai"), patch.multiple(
            aux, _read_main_provider=lambda: "custom", _read_main_model=lambda: "m",
            _resolve_custom_runtime=lambda: (None, None, None), _is_provider_unhealthy=lambda *_a, **_k: False,
            main_model_fallback_refused=lambda *_a: False),
            patch.object(aux, "resolve_provider_client", return_value=(client, "glm-resolved"))]
        got = _run(lambda: aux._try_main_agent_model_fallback("zai", "compression", failed_model="glm"), *marks)
        assert got[:2] == (client, "glm-resolved")
        assert seen == [({"provider": "custom", "model": "m"}, False),
                        ({"provider": "zai", "model": "glm-resolved", "slot_provider": "custom"}, True)]


DEEPSEEK = "https://api.deepseek.com/v1"


def _reasoner_only(seen):
    def custody(candidate, required, **_kw):
        seen.append((dict(candidate), required))
        return {"action": "allow" if candidate.get("model") == "deepseek-reasoner" else "skip"}
    return [custody]


class TestNameMarkedRewrittenModel:
    """A route the mark names is judged on its configured model before the build. Its provider can send
    another model, so the walk's main step and the last-resort hop judge the built client again on the
    model it sends. The gate admits ``deepseek-reasoner`` only."""

    @staticmethod
    def _resolves_to(client, model):
        real = aux.resolve_provider_client

        def resolve(provider, *args, **kwargs):
            return (client, model) if provider == "deepseek" else real(provider, *args, **kwargs)
        return patch.object(aux, "resolve_provider_client", side_effect=resolve)

    @pytest.mark.parametrize("sends,served", [("deepseek-flash", False), ("deepseek-reasoner", True)],
                             ids=["rewritten", "as-judged"])
    def test_the_walk_judges_the_model_its_main_step_sends(self, hooks, sends, served):
        seen, built = [], _client(DEEPSEEK)
        hooks["route_custody"] = _reasoner_only(seen)
        marks = [*_marked("deepseek"), patch.multiple(
            aux, _main_route_target=lambda _runtime, _task: ("deepseek", "deepseek-reasoner", "", "", ""),
            _is_provider_unhealthy=lambda *_a, **_k: False, _warn_stale_openai_base_url=lambda *_a, **_k: None,
            _get_auxiliary_task_config=lambda *_a, **_k: {}, _read_main_provider=lambda: "deepseek"),
            patch("hermes_cli.config.load_config_readonly", return_value={}), self._resolves_to(built, sends)]
        got = _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)
        assert got == ((built, sends) if served else (None, None))
        assert [candidate["model"] for candidate, _required in seen] == list(dict.fromkeys(["deepseek-reasoner", sends]))

    @pytest.mark.parametrize("sends,served", [("deepseek-flash", False), ("deepseek-reasoner", True)],
                             ids=["rewritten", "as-judged"])
    def test_the_hop_judges_the_model_its_client_sends(self, hooks, sends, served):
        seen, client = [], _client(DEEPSEEK)
        hooks["route_custody"] = _reasoner_only(seen)
        route_hooks._bind_scope(SimpleNamespace(provider="deepseek", model="deepseek-reasoner", base_url="",
                                                session_id="s-r", platform="", _provider_fallback_active=False,
                                                _route_resolution_entry=None, _fallback_chain=[]))
        marks = [*_marked("deepseek"), patch.multiple(
            aux, _read_main_provider=lambda: "deepseek", _read_main_model=lambda: "deepseek-reasoner",
            _is_provider_unhealthy=lambda *_a, **_k: False, main_model_fallback_refused=lambda *_a: False),
            patch.object(aux, "resolve_provider_client", return_value=(client, sends))]
        got = _run(lambda: aux._try_main_agent_model_fallback("zai", "compression", failed_model="glm"), *marks)
        assert got[:2] == ((client, sends) if served else (None, None))
        assert [candidate["model"] for candidate, _required in seen] == list(dict.fromkeys(["deepseek-reasoner", sends]))


# ``deepseek-reasoner`` is a retired alias that the real normalizer folds onto ``deepseek-flash``;
# ``deepseek-v4-pro`` goes to the wire as configured.
FOLDS = [("deepseek-reasoner", "deepseek-flash", False), ("deepseek-v4-pro", "deepseek-v4-pro", True)]
FOLD_IDS = ["folded", "nothing-folds"]


def _admits(seen, configured):
    """A gate that admits only the *configured* model, and records each candidate."""
    def custody(candidate, required, **_kw):
        seen.append((dict(candidate), required))
        return {"action": "allow" if candidate.get("model") == configured else "skip"}
    return [custody]


def _entry_walk(entry, client, *, chain):
    """The ``auto`` walk with no main client and *entry* on the main (``main``) or the task (``task``)
    fallback chain. The entry builds *client* on the model the real normalizer resolves for it."""
    main_chain = {"fallback_providers": [entry]} if chain == "main" else {}
    task_config = {"fallback_chain": [entry]} if chain == "task" else {}
    return [patch.multiple(
        aux, _main_route_target=lambda _runtime, _task: ("zai", "glm", "", "", ""),
        _try_main_provider_route=lambda *_a: None, _read_main_provider=lambda: "zai",
        _warn_stale_openai_base_url=lambda *_a, **_k: None, _get_auxiliary_task_config=lambda *_a, **_k: task_config,
        _is_provider_unhealthy=lambda *_a, **_k: False, _context_too_small=lambda *_a, **_k: None,
        _resolve_fallback_entry=lambda e: (client, aux._normalize_resolved_model(e["model"], e["provider"]))),
        patch("hermes_cli.config.load_config_readonly", return_value=main_chain)]


class TestChainEntryRewrittenModel:
    """A fallback chain entry is judged on its configured model before the build. When its provider folds
    that model, the walk judges the entry again on the model its client sends: as the marked route when the
    mark names the entry, and by a loaded gate with no mark. A cache hit is judged the same way."""

    @pytest.mark.parametrize("chain", ["main", "task"])
    @pytest.mark.parametrize("configured,sends,served", FOLDS, ids=FOLD_IDS)
    def test_a_marked_entry_is_judged_on_the_model_it_sends(self, hooks, chain, configured, sends, served):
        seen, built = [], _client(DEEPSEEK)
        hooks["route_custody"] = _admits(seen, configured)
        marks = [*_marked("deepseek"), *_entry_walk({"provider": "deepseek", "model": configured}, built, chain=chain)]
        got = _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)
        assert got == ((built, sends) if served else (None, None))
        assert [(c["provider"], c["model"], required) for c, required in seen] == [
            ("deepseek", model, True) for model in dict.fromkeys([configured, sends])]
        built.chat.completions.create.assert_not_called()

    @pytest.mark.parametrize("configured,sends,served", FOLDS, ids=FOLD_IDS)
    def test_with_no_mark_a_loaded_gate_judges_the_model_it_sends(self, hooks, configured, sends, served):
        seen, built = [], _client(DEEPSEEK)
        hooks["route_custody"] = _admits(seen, configured)
        marks = [*_marked("deepseek", False), *_entry_walk({"provider": "deepseek", "model": configured}, built, chain="main")]
        got = _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)
        assert got == ((built, sends) if served else (None, None))
        assert [(c["model"], required) for c, required in seen] == [(m, False) for m in dict.fromkeys([configured, sends])]

    def test_with_no_mark_and_no_gate_the_folded_entry_is_served(self, hooks):
        built = _client(DEEPSEEK)
        marks = [*_marked("deepseek", False), *_entry_walk({"provider": "deepseek", "model": "deepseek-reasoner"}, built,
                                                          chain="main")]
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks) == (built, "deepseek-flash")

    def test_a_cache_hit_is_judged_on_the_model_it_sends(self, hooks):
        """The gate admits the folded model on the walk and refuses it on the hit, with no mark."""
        seen, built, refuse = [], _client(DEEPSEEK), []

        def custody(candidate, required, **_kw):
            seen.append(candidate["model"])
            return {"action": "skip" if refuse and candidate["model"] == "deepseek-flash" else "allow"}
        hooks["route_custody"] = [custody]
        marks = [*_marked("deepseek", False), *_entry_walk({"provider": "deepseek", "model": "deepseek-reasoner"}, built,
                                                          chain="main")]
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks)[0] is built
        refuse.append(True)
        assert _run(lambda: aux._get_cached_client("auto", None, task="compression"), *marks) == (None, None)
        assert seen[:4] == ["deepseek-reasoner", "deepseek-flash", "deepseek-reasoner", "deepseek-flash"]


def _deepseek_hop(client, sends):
    return [patch.multiple(
        aux, _read_main_provider=lambda: "deepseek", _read_main_model=lambda: "deepseek-reasoner",
        _is_provider_unhealthy=lambda *_a, **_k: False, main_model_fallback_refused=lambda *_a: False),
        patch.object(aux, "resolve_provider_client", return_value=(client, sends))]


class TestUnmarkedRewrittenModel:
    """With no mark, a loaded gate judges a rewritten model on the walk's main step and the hop, as on a
    chain entry. With no gate and no mark nothing is asked and the client is served as before."""

    @pytest.mark.parametrize("sends,served", [("deepseek-flash", False), ("deepseek-reasoner", True)],
                             ids=["rewritten", "as-judged"])
    def test_the_hop_judges_the_model_its_client_sends(self, hooks, sends, served):
        seen, client = [], _client(DEEPSEEK)
        hooks["route_custody"] = _reasoner_only(seen)
        marks = [*_marked("deepseek", False), *_deepseek_hop(client, sends)]
        got = _run(lambda: aux._try_main_agent_model_fallback("zai", "compression", failed_model="glm"), *marks)
        assert got[:2] == ((client, sends) if served else (None, None))
        assert [(c["provider"], c["model"], required) for c, required in seen] == [
            ("deepseek", model, False) for model in dict.fromkeys(["deepseek-reasoner", sends])]

    def test_with_no_gate_the_hop_sends_as_before(self, hooks):
        client = _client(DEEPSEEK)
        marks = [*_marked("deepseek", False), *_deepseek_hop(client, "deepseek-flash")]
        got = _run(lambda: aux._try_main_agent_model_fallback("zai", "compression", failed_model="glm"), *marks)
        assert got[:2] == (client, "deepseek-flash")

    @pytest.mark.parametrize("sends,served", [("deepseek-flash", False), ("deepseek-reasoner", True)],
                             ids=["rewritten", "as-configured"])
    def test_the_walk_judges_the_model_its_main_step_sends(self, hooks, sends, served):
        seen, built = [], _client(DEEPSEEK)
        hooks["route_custody"] = _reasoner_only(seen)
        marks = [*_marked("deepseek", False), patch.multiple(
            aux, _main_route_target=lambda _runtime, _task: ("deepseek", "deepseek-reasoner", "", "", ""),
            _try_main_provider_route=lambda *_a: (built, sends, "deepseek"), _discovery_chain_allowed=lambda *_a: False,
            _warn_stale_openai_base_url=lambda *_a, **_k: None, _get_auxiliary_task_config=lambda *_a, **_k: {}),
            patch("hermes_cli.config.load_config_readonly", return_value={})]
        got = _run(lambda: aux._resolve_auto_route(task="compression"), *marks)
        assert got[:2] == ((built, sends) if served else (None, None))
        assert [c["model"] for c, _required in seen] == ([] if served else [sends]), "only a rewritten model is asked"


class TestTextSlotRewrittenModel:
    """``get_text_auxiliary_client``, which kanban's ``goal_judge`` handoff check calls, judges an explicit
    slot on its configured model before the build and again on the model its client sends."""

    @staticmethod
    def _slot(client, sends):
        return [patch.object(aux, "_resolve_task_provider_model",
                             return_value=("deepseek", "deepseek-reasoner", None, None, None)),
                patch.object(aux, "resolve_provider_client", return_value=(client, sends))]

    @pytest.mark.parametrize("sends,served", [("deepseek-flash", False), ("deepseek-reasoner", True)],
                             ids=["rewritten", "as-judged"])
    def test_a_marked_slot_is_judged_on_the_model_it_sends(self, hooks, sends, served):
        seen, client = [], _client(DEEPSEEK)
        hooks["route_custody"] = _reasoner_only(seen)
        got = _run(lambda: aux.get_text_auxiliary_client("goal_judge"), *_marked("deepseek"), *self._slot(client, sends))
        assert got == ((client, sends) if served else (None, None))
        assert [c["model"] for c, _required in seen] == list(dict.fromkeys(["deepseek-reasoner", sends]))

    @pytest.mark.parametrize("sends,served", [("deepseek-flash", False), ("deepseek-reasoner", True)],
                             ids=["rewritten", "as-configured"])
    def test_with_no_mark_a_loaded_gate_judges_a_rewritten_model(self, hooks, sends, served):
        seen, client = [], _client(DEEPSEEK)
        hooks["route_custody"] = _reasoner_only(seen)
        got = _run(lambda: aux.get_text_auxiliary_client("goal_judge"), *_marked("deepseek", False), *self._slot(client, sends))
        assert got == ((client, sends) if served else (None, None))
        assert [c["model"] for c, _required in seen] == ([] if served else [sends])


BAD = "http://[::1/v1"


def _agent_at(url, *, kept=False):
    record = route_hooks.resolution_record({"provider": "custom", "model": "m", "base_url": LOCAL}, {"provider": "custom"})
    return SimpleNamespace(provider="custom", model="m", base_url=url, session_id="s-bad", platform="",
                           _provider_fallback_active=kept, _route_fallback_entry={"provider": "custom", "model": "m"},
                           _route_resolution_entry=record, _fallback_chain=[], _rate_limited_until=0,
                           _restore_primary_runtime=lambda: False)


class TestUnparsedAgentUrl:
    """An agent URL that ``urlsplit`` rejects: inert with no gate and no mark, an allow under either."""

    @pytest.mark.parametrize("kept", [False, True], ids=["resolution-entry", "kept-fallback"])
    def test_with_no_gate_and_no_mark_nothing_applies(self, hooks, kept):
        agent = _agent_at(BAD, kept=kept)
        marks = _marked("zai", False)
        assert _run(lambda: route_hooks._resolution_fallback_entry(agent), *marks) is None
        record = _run(lambda: route_hooks.side_route_record(agent), *marks)
        assert record is None or not route_hooks.custody_applies(record)
        assert not _run(lambda: route_hooks.agent_custody_applies(agent), *marks)

    def test_the_inert_hop_builds_as_before(self, hooks):
        route_hooks._bind_scope(_agent_at(BAD))
        client = _client(LOCAL)
        marks = [*_marked("zai", False), patch.multiple(
            aux, _read_main_provider=lambda: "custom", _read_main_model=lambda: "m",
            _resolve_custom_runtime=lambda: (LOCAL, "key", None), _is_provider_unhealthy=lambda *_a, **_k: False,
            main_model_fallback_refused=lambda *_a: False), patch.object(aux, "resolve_provider_client",
                                                                          return_value=(client, "m"))]
        assert _run(lambda: aux._try_main_agent_model_fallback("zai", "compression", failed_model="glm"), *marks)[0] is client

    @pytest.mark.parametrize("kept", [False, True], ids=["resolution-entry", "kept-fallback"])
    def test_under_a_mark_the_route_sends_nothing(self, hooks, kept):
        agent = _agent_at(BAD, kept=kept)
        assert _run(lambda: route_hooks.agent_custody_applies(agent), *_marked("zai"))
        with pytest.raises(route_hooks.RouteCustodyRefused):
            _run(lambda: route_hooks.check_request_route(agent, [{"role": "user", "content": "q"}]), *_marked("zai"))

    def test_under_a_gate_the_entry_needs_an_allow(self, hooks):
        seen = []
        hooks["route_custody"] = _gate(seen)
        agent = _agent_at(BAD)
        entry = _run(lambda: route_hooks._resolution_fallback_entry(agent), *_marked("zai", False))
        assert route_hooks.entry_requires_custody(entry)
        assert "base_url" not in entry or entry["base_url"] == LOCAL, "the unparsed URL was kept"
        assert not _run(lambda: route_hooks.check_request_route(agent, []), *_marked("zai", False))
        assert [required for _candidate, required in seen] == [True]

    @pytest.mark.parametrize("off", [None, "off", False], ids=["unset", "off-string", "false"])
    def test_under_a_mark_alone_the_record_needs_an_allow(self, hooks, off):
        """No gate is loaded. The mark alone makes an entry at an unparsed URL required, over its own key."""
        entry = {"provider": "custom", "model": "m", "base_url": BAD, "route_custody": off}
        record = _run(lambda: route_hooks.resolution_record(entry, {"provider": "custom", "base_url": BAD}),
                      *_marked("zai"))
        assert record[route_hooks.ROUTE_CUSTODY_KEY] == route_hooks.ROUTE_CUSTODY_REQUIRED

    def test_a_delegated_child_record_under_a_gate_needs_an_allow(self, hooks):
        hooks["route_custody"] = _gate([])
        record = _run(lambda: route_hooks.side_route_record(_agent_at(BAD, kept=True)), *_marked("zai", False))
        assert route_hooks.entry_requires_custody(record)
        assert "resolved_base_url" not in record


ZAI_RUNTIME = {"provider": "custom", "model": "m", "base_url": ZAI}


def _off_mark_parent():
    return SimpleNamespace(provider="custom", model="m", base_url=LOCAL, session_id="s-p", platform="",
                           _provider_fallback_active=False, _route_resolution_entry=None, _fallback_chain=[])


class TestSideQuestionRuntimeAtTheMark:
    """A side question whose parent runs off the mark, on a runtime at the marked endpoint."""

    def test_the_runtime_is_judged_before_the_fork(self, hooks):
        from agent import side_question
        fork, oneshot = MagicMock(return_value="forked"), MagicMock(return_value="answer")
        with patch.object(side_question, "_answer_via_fork", fork), \
                patch.object(side_question, "_answer_via_oneshot", oneshot), \
                patch.object(side_question, "_side_question_task_config", return_value={}):
            with pytest.raises(route_hooks.RouteCustodyRefused):
                _run(lambda: side_question.answer_side_question(
                    "q", [], parent_agent=_off_mark_parent(), main_runtime=dict(ZAI_RUNTIME),
                    route={"record": None, "session_id": "s-p"}), *_marked("zai"))
        fork.assert_not_called()
        oneshot.assert_not_called()


def _vision(runtime, *marks):
    from gateway import route_vision
    enriched = []

    async def enrich():
        enriched.append(1)
        return "described"
    out = _run(lambda: asyncio.run(route_vision.vision_under_custody(
        enrich, None, runtime, session_id="s-v", messages=[], message_text="hi", image_paths=["/img/a.png"])), *marks)
    return out, enriched


class TestVisionRuntimeAtTheMark:
    """Vision enrichment judges the runtime at its ``base_url``, as a side question does."""

    def test_with_no_gate_no_image_leaves(self, hooks):
        with patch.object(route_hooks, "vision_rides_main_route", return_value=True):
            out, enriched = _vision(dict(ZAI_RUNTIME), *_marked("zai"))
        assert enriched == []
        assert "/img/a.png" in out

    def test_the_gate_sees_the_runtime_as_the_marked_route(self, hooks):
        seen = []
        hooks["route_custody"] = _gate(seen, refuse={"zai"})
        with patch.object(route_hooks, "vision_rides_main_route", return_value=True):
            out, enriched = _vision(dict(ZAI_RUNTIME), *_marked("zai"))
        assert enriched == [] and "/img/a.png" in out
        assert ({"provider": "zai", "model": "m", "slot_provider": "custom"}, True) in seen

    def test_a_runtime_elsewhere_enriches_as_before(self, hooks):
        with patch.object(route_hooks, "vision_rides_main_route", return_value=True):
            out, enriched = _vision({**ZAI_RUNTIME, "base_url": LOCAL}, *_marked("zai"))
        assert (out, enriched) == ("described", [1])


class _Compressor:
    built = []

    def __init__(self, *_args, **_kwargs):
        _Compressor.built.append(self)

    def compress(self, messages, **_kw):
        return messages


def _mutate_compress(runtime, *marks):
    from gateway.session_mutation_compress import prepare_compress
    authority = MagicMock()
    authority.runner._resolve_session_agent_runtime.return_value = ("m", dict(runtime))
    authority.runner._cached_agent_for = lambda _key: None
    authority.db.get_messages_as_conversation.return_value = [{"role": "user", "content": "q"},
                                                             {"role": "assistant", "content": "a"}]
    authority.db.get_session.return_value = None
    policy = SimpleNamespace(config=lambda _authority: {}, safe_mode=True)
    prepared = {"snapshot": {"target": "s-c", "receipt": {"policy": {}}}}
    _Compressor.built.clear()
    return _run(lambda: asyncio.run(prepare_compress(authority, SimpleNamespace(source=None, route="key"), {}, prepared)),
                *marks, patch("gateway.session_policy.restore_policy", return_value=policy),
                patch("gateway.session_policy.policy_scope", lambda *_a, **_kw: contextlib.nullcontext()),
                patch("agent.context_compressor.ContextCompressor", _Compressor))


class TestMutateCompressRuntimeAtTheMark:
    """``session.mutate compress`` judges the runtime it summarizes on at its ``base_url``."""

    def test_with_no_gate_nothing_is_summarized(self, hooks):
        from hermes_state_runtime import RuntimeStoreError
        with pytest.raises(RuntimeStoreError) as refused:
            _mutate_compress(ZAI_RUNTIME, *_marked("zai"))
        assert refused.value.reason == "route_custody_refused"
        assert _Compressor.built == []

    def test_a_runtime_elsewhere_is_summarized_as_before(self, hooks):
        assert _mutate_compress({**ZAI_RUNTIME, "base_url": LOCAL}, *_marked("zai"))["status"] == "noop"
        assert len(_Compressor.built) == 1
