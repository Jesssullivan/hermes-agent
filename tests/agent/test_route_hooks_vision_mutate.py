"""Route custody for gateway vision enrichment and a rotating ``session.mutate compress`` (agent/route_hooks.py).

The fourth review of TIN-7765 found these paths open at a1dd6173:

V: gateway vision enrichment sent the user's image on a resolution-time entry before custody saw
   the turn; ``/background`` ran the same enrichment with no scoped runtime.
C2: a rotating ``session.mutate compress`` minted a child id and left the parent's labels and
    data use behind.
End of turn: the route scope stayed bound after a turn, so a later compaction in the same context
    skipped the ``compaction`` phase check.

Each probe drives the production code with stub agents and patched resolvers. No provider is
called."""

import asyncio
import contextlib
import contextvars
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import route_hooks
from tests.agent.test_route_hooks_out_of_turn import (
    HOSTED, RECORD, _compression_seam, _fallback_runtime, _hooks_md_gate, _label_in_its_own_context,
    _make_agent, _prepare_compress, _resolve_allows_compaction_refuses, _SummaryRecorder, _tool_message,
    _transcript)
from hermes_cli.plugins import PluginManager


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


def _hooks_md_example_gate(hooks, seen):
    """The ``hooks.md`` example gate, with ``openai`` as the hosted route: labels or data use refuse it."""
    hooks["turn_route"] = [lambda data_use, **_kw: {"labels": ["acme:client"]} if "acme:client-raw" in data_use else None]

    def custody(phase, candidate, labels, data_use, session_id, task, **_kw):
        seen.append((phase, candidate["provider"], labels, data_use, session_id, task))
        refused = ("acme:client" in labels or "acme:client-raw" in data_use) and candidate["provider"] == "openai"
        return {"action": "skip", "reason": "client data"} if refused else {"action": "allow"}

    hooks["route_custody"] = [custody]


# -- V: gateway vision enrichment -----------------------------------------------------------------


def _vision_runner(record, *, session_id="vision-session"):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")})
    runner.adapters = {}
    runner._decide_image_input_mode = lambda **_: "text"
    runner.session_store = MagicMock()
    runner.session_store.peek_session_id.return_value = session_id
    runner._resolve_session_key_or_none = lambda _source, key=None: key or "tg:1"

    def resolve(**_kw):  # the gateway resolver's contract: stash the entry record in this context
        route_hooks.stash_resolution_entry(record)
        return "gpt-4o", {"provider": "openrouter", "api_key": "k"}

    runner._resolve_session_agent_runtime = resolve
    return runner


def _image_event():
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="private", user_name="u")
    return MessageEvent(text="what is this", message_type=MessageType.PHOTO, source=source,
                        media_urls=["/cache/img_1.png"], media_types=["image/png"])


class _VisionRecorder:
    """Stands in for ``vision_analyze_tool``: records the main route the image would ride."""

    def __init__(self):
        self.calls = []

    async def __call__(self, image_url, user_prompt=None, **_kw):
        from agent.auxiliary_client import _runtime_main_value
        self.calls.append((image_url, _runtime_main_value("provider")))
        return json.dumps({"success": True, "analysis": "a chart"})


def _prepare(runner, history):
    event, recorder = _image_event(), _VisionRecorder()
    with patch("tools.vision_tools.vision_analyze_tool", recorder):
        text = asyncio.run(runner._prepare_inbound_message_text(
            event=event, source=event.source, history=history, session_key="tg:1"))
    return text, recorder


class TestVisionEnrichment:
    def test_a_gate_that_admits_only_at_resolve_keeps_the_image_off_the_entry(self, hooks):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        text, recorder = _prepare(_vision_runner(dict(RECORD)), [])
        assert recorder.calls == []  # no pixels left for the refused entry
        assert asked == [("auxiliary", "openai")]
        assert "/cache/img_1.png" in text and "what is this" in text

    def test_the_hooks_md_gate_sees_mid_turn_tool_data_of_the_session(self, hooks):
        seen = []
        _hooks_md_example_gate(hooks, seen)
        record = route_hooks.resolution_record(HOSTED, {"provider": "openrouter"})  # a plain hosted entry
        _text, recorder = _prepare(_vision_runner(record), _transcript())
        assert recorder.calls == []
        assert seen == [("auxiliary", "openai", [], ["acme:client-raw"], "vision-session", "vision")]

    def test_an_admitted_entry_enriches_on_the_sessions_runtime(self, hooks):
        seen = []
        _hooks_md_example_gate(hooks, seen)
        record = route_hooks.resolution_record(HOSTED, {"provider": "openrouter"})
        text, recorder = _prepare(_vision_runner(record), [{"role": "user", "content": "hi"}])
        assert recorder.calls == [("/cache/img_1.png", "openrouter")]
        assert seen[0][0] == "auxiliary" and seen[0][5] == "vision"
        assert "a chart" in text

    def test_with_no_hook_a_plain_entry_enriches_as_before(self):
        record = route_hooks.resolution_record(HOSTED, {"provider": "openrouter"})
        _text, recorder = _prepare(_vision_runner(record), _transcript())
        assert recorder.calls == [("/cache/img_1.png", "openrouter")]

    def test_an_explicit_vision_slot_does_not_ride_the_main_route(self, hooks):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("deepinfra", "vision-model", None, None, None)):
            _text, recorder = _prepare(_vision_runner(dict(RECORD)), [])
        assert len(recorder.calls) == 1
        assert asked == []


def _background_runner():
    from gateway.hooks import HookRegistry
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters, runner._voice_mode, runner._session_db, runner._reasoning_config = {}, {}, None, None
    runner._provider_routing, runner._fallback_model, runner._running_agents = {}, None, {}
    runner._background_tasks = set()
    runner.session_store = MagicMock()
    runner.session_store.get_model_override.return_value = None
    runner.hooks = HookRegistry()
    return runner


def _run_background(runner):
    from gateway.config import Platform
    from gateway.session import SessionSource

    adapter = AsyncMock()
    adapter.extract_media = MagicMock(return_value=([], "done"))
    adapter.extract_images = MagicMock(return_value=([], "done"))
    runner.adapters[Platform.TELEGRAM] = adapter
    seen_runtime = []

    async def enrich(prompt, _paths):
        from agent.auxiliary_client import _runtime_main_value
        seen_runtime.append(_runtime_main_value("provider"))
        return "described: " + prompt

    runner._enrich_message_with_vision = enrich
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="2", user_name="u")
    built = MagicMock()
    built.run_conversation.return_value = {"final_response": "done", "messages": []}
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=_fallback_runtime()), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("run_agent.AIAgent", return_value=built):
        asyncio.run(runner._run_background_task("look", source, "bg_1", media_urls=["/cache/img_2.png"],
                                                media_types=["image/png"]))
    return seen_runtime, built


class TestBackgroundVision:
    def test_a_refused_entry_sends_no_image(self, hooks):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        seen_runtime, _built = _run_background(_background_runner())
        assert seen_runtime == []
        assert asked[0] == ("auxiliary", "openai")

    def test_an_admitted_entry_enriches_on_the_tasks_runtime_not_the_mirrors(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        seen_runtime, built = _run_background(_background_runner())
        assert seen_runtime == ["openrouter"]
        assert "described: look" in built.run_conversation.call_args.kwargs["user_message"]


# -- C2: a rotating session.mutate compress -------------------------------------------------------


def _write_rotating_compress(parent, child):
    from gateway import session_mutations

    result = {"target_session_id": child, "previous_target_session_id": parent, "execution_generation": 2,
              "message_count": 1, "summary": None}

    def mutate(_db, **fields):
        if fields.get("_prepare_only"):
            return {"snapshot": {"target": parent}}
        fields["_live_guard"](set())
        return result

    async def prepare(_authority, _live, _payload, prepared):
        return dict(prepared, messages=[{"role": "user", "content": "summary"}], post_compaction=None, in_place=False)

    authority = MagicMock()
    authority.sessions = {}
    authority.db.get_session.return_value = None
    ref = SimpleNamespace(session_id="logical")
    params = {"operation": "compress", "request_id": "r1", "expected_revision": 1, "expected_generation": 1,
              "payload": {}}
    with patch.object(session_mutations, "mutate_runtime_session", side_effect=mutate), \
         patch("gateway.session_mutation_compress.prepare_compress", prepare):
        return asyncio.run(session_mutations.write_mutation(authority, ref, params, None, principal_id="p"))


class TestRotatingSessionMutate:
    def test_the_child_turn_keeps_the_parents_labels_and_data_use(self, hooks):
        seen = []
        _hooks_md_gate(hooks, seen)  # labels only, so the carried labels alone must refuse
        _label_in_its_own_context("mutate-parent")
        _write_rotating_compress("mutate-parent", "mutate-child")
        assert route_hooks.session_labels("mutate-child") == {"acme:client"}
        assert "acme:client-raw" in route_hooks.session_data_use("mutate-child")
        agent = _make_agent(session_id="mutate-child")
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        with pytest.raises(route_hooks.RouteCustodyRefused):
            route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "summary"}], turn_id="t2")

    def test_the_summary_keeps_the_transcript_data_use_of_a_session_with_no_turn_here(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        _prepare_compress(None, _transcript(), "cold-parent")  # a restart: no turn of it ran in this process
        _write_rotating_compress("cold-parent", "cold-child")
        assert "acme:client-raw" in route_hooks.session_data_use("cold-child")

    def test_in_place_compress_keeps_the_session_state(self, hooks):
        _hooks_md_gate(hooks, [])
        _label_in_its_own_context("in-place")
        _write_rotating_compress("in-place", "in-place")
        assert route_hooks.session_labels("in-place") == {"acme:client"}


class TestAgentCompactionDataUse:
    def test_a_mid_turn_stamp_survives_the_summary(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        agent = _make_agent(session_id="mid-turn")
        route_hooks.begin_turn(agent, messages=[{"role": "user", "content": "q"}], turn_id="t1")
        with _compression_seam(_SummaryRecorder()):
            agent._compress_context(_transcript(), "")  # the tool row arrived after turn start
        assert "acme:client-raw" in route_hooks.session_data_use("mid-turn")


# -- End of turn ----------------------------------------------------------------------------------


def _run_turn(agent):
    def turn(agent_, *_a, **_kw):
        route_hooks.begin_turn(agent_, messages=[_tool_message(["acme:client-raw"])], turn_id="t1")
        return {"final_response": "ok", "messages": []}

    from agent.conversation_loop import run_conversation
    with patch("agent.conversation_loop._run_conversation_turn", turn):
        return run_conversation(agent, "hello")


class TestEndOfTurn:
    def test_a_compaction_after_the_turn_in_the_same_context_is_checked(self, hooks):
        asked, recorder = [], _SummaryRecorder()
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        agent = _make_agent(session_id="same-context")
        route_hooks.bind_resolution_entry(agent, dict(RECORD))
        with patch("agent.route_hooks._recheck_resolution_fallback"):  # admit the turn itself
            _run_turn(agent)
        with _compression_seam(recorder), pytest.raises(route_hooks.RouteCustodyRefused):
            agent._compress_context(_transcript(), "")
        assert recorder.calls == 0
        assert ("compaction", "openai") in asked

    def test_an_auxiliary_call_after_the_turn_still_sees_the_session(self, hooks):
        seen = []
        _hooks_md_gate(hooks, seen)
        agent = _make_agent(session_id="after-turn")
        _run_turn(agent)
        assert route_hooks.auxiliary_refuses(dict(RECORD), task="title_generation", failed_provider="anthropic")
        assert seen[-1][2] == ["acme:client"]


class TestPhaseAdvertisement:
    def test_every_custody_phase_is_listed(self):
        assert route_hooks.HERMES_ROUTE_CUSTODY_PHASES == frozenset(
            {"failover", "resolve", "turn", "auxiliary", "compaction"})
