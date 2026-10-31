"""Route custody on the remaining vision paths before a turn (agent/route_hooks.py).

The review of c9ea31be found these paths open:

Main slot: ``auxiliary.vision.provider: main`` rides the session's main route, and the vision check
    skipped it because only ``auto`` counted as the main route.
CLI: text-mode image pre-analysis (``_preprocess_images_with_vision``) sent the image on the agent's
    kept fallback or resolution-time entry with no custody check.
Inbound labels: the vision check saw only earlier turns' labels, never ``turn_route``'s labels for
    the inbound message.
Sticker: Telegram sticker vision ran with no session route and no custody check.
Carry: a rotating ``session.mutate compress`` read only the process registry, not the resident
    agent's held copies.

Each probe drives the production code with stub agents and patched resolvers. No provider is
called."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import route_hooks
from hermes_cli.plugins import PluginManager
from tests.agent.test_route_hooks_out_of_turn import (
    HOSTED, RECORD, REQUIRED, _make_agent, _resolve_allows_compaction_refuses, _tool_message, _transcript)
from tests.agent.test_route_hooks_vision_mutate import (
    _VisionRecorder, _hooks_md_example_gate, _prepare, _vision_runner, _write_rotating_compress)


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


def _vision_slot(provider, base_url=""):
    slot = {"provider": provider, **({"base_url": base_url} if base_url else {})}
    return patch("agent.auxiliary_client._get_auxiliary_task_config",
                 side_effect=lambda task: slot if task == "vision" else {})


# -- Main slot: provider main rides the main route ------------------------------------------------


class TestMainVisionSlot:
    @pytest.mark.parametrize("provider", ["main", "custom:main", "openrouter", "auto"])
    def test_a_slot_on_the_main_route_rides_it(self, provider):
        from agent.auxiliary_client import scoped_runtime_main
        with _vision_slot(provider), scoped_runtime_main({"provider": "openrouter", "model": "gpt-4o"}):
            assert route_hooks.vision_rides_main_route() is True
            assert route_hooks.vision_rides_main_route({"provider": "openrouter"}) is True

    @pytest.mark.parametrize("provider,base_url", [("deepinfra", ""), ("main", "https://vision.example/v1")])
    def test_a_named_provider_or_an_endpoint_does_not(self, provider, base_url):
        from agent.auxiliary_client import scoped_runtime_main
        with _vision_slot(provider, base_url), scoped_runtime_main({"provider": "openrouter", "model": "gpt-4o"}):
            assert route_hooks.vision_rides_main_route({"provider": "openrouter"}) is False

    def test_gateway_enrichment_on_a_main_slot_checks_the_entry(self, hooks):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        with _vision_slot("main"):
            text, recorder = _prepare(_vision_runner(dict(RECORD)), [])
        assert recorder.calls == []
        assert asked == [("auxiliary", "openai")]
        assert "/cache/img_1.png" in text


# -- CLI: text-mode image pre-analysis ------------------------------------------------------------


def _cli(agent, history=None):
    from cli import HermesCLI
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._attached_images, cli_obj._image_counter = [], 0
    cli_obj.agent, cli_obj.conversation_history = agent, list(history or [])
    return cli_obj


def _cli_preprocess(cli_obj, tmp_path, text="check this"):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    recorder = _VisionRecorder()
    with patch("tools.vision_tools.vision_analyze_tool", recorder):
        result = cli_obj._preprocess_images_with_vision(text, [img], announce=False)
    return result, recorder, img


def _cli_agent(session_id="cli-session"):
    agent = _make_agent(session_id=session_id)
    route_hooks.bind_resolution_entry(agent, dict(RECORD), source="init")
    return agent


class TestCliVision:
    def test_a_refused_resolution_entry_sends_no_image(self, hooks, tmp_path):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        result, recorder, img = _cli_preprocess(_cli(_cli_agent()), tmp_path)
        assert recorder.calls == []
        assert asked == [("auxiliary", "openai")]
        assert str(img) in result and "check this" in result

    def test_a_refused_kept_fallback_sends_no_image(self, hooks, tmp_path):
        asked = []
        hooks["route_custody"] = [_resolve_allows_compaction_refuses(asked)]
        agent = _make_agent(session_id="cli-kept")
        agent._provider_fallback_active = True
        agent._route_fallback_entry = {"provider": "openrouter", "model": "gpt-4o"}
        result, recorder, img = _cli_preprocess(_cli(agent), tmp_path)
        assert recorder.calls == []
        assert asked == [("auxiliary", "openrouter")]
        assert str(img) in result

    def test_the_hooks_md_gate_sees_the_cli_transcript_data_use(self, hooks, tmp_path):
        seen = []
        _hooks_md_example_gate(hooks, seen)
        agent = _make_agent(session_id="cli-data")
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(HOSTED, {"provider": "openrouter"}))
        _result, recorder, _img = _cli_preprocess(_cli(agent, _transcript()), tmp_path)
        assert recorder.calls == []
        assert seen[-1][0] == "auxiliary" and "acme:client-raw" in seen[-1][3] and seen[-1][4] == "cli-data"

    def test_an_admitted_entry_enriches_on_the_agents_route(self, hooks, tmp_path):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        result, recorder, _img = _cli_preprocess(_cli(_cli_agent()), tmp_path)
        assert [provider for _url, provider in recorder.calls] == ["openrouter"]
        assert "a chart" in result

    def test_with_no_hook_a_plain_entry_enriches_as_before(self, tmp_path):
        agent = _make_agent(session_id="cli-plain")
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(HOSTED, {"provider": "openrouter"}))
        result, recorder, _img = _cli_preprocess(_cli(agent, _transcript()), tmp_path)
        assert len(recorder.calls) == 1
        assert "a chart" in result

    def test_with_no_agent_enrichment_runs_as_before(self, hooks, tmp_path):
        hooks["route_custody"] = [lambda **_kw: {"action": "skip"}]
        _result, recorder, _img = _cli_preprocess(_cli(None), tmp_path)
        assert len(recorder.calls) == 1


# -- Inbound labels: turn_route sees the inbound message before the vision check -----------------


def _content_labeller(hooks, gate_calls, *, raises=False):
    def turn_route(phase, messages, **_kw):
        if raises:
            raise RuntimeError("labeller down")
        if any("acme" in str(m.get("content") or "") for m in messages):
            return {"labels": ["acme:client"]}
        return None

    def custody(phase, labels, **_kw):
        gate_calls.append((phase, labels))
        return {"action": "skip"} if "acme:client" in labels else {"action": "allow"}

    hooks["turn_route"], hooks["route_custody"] = [turn_route], [custody]


def _prepare_text(runner, text, history=()):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="private", user_name="u")
    event = MessageEvent(text=text, message_type=MessageType.PHOTO, source=source,
                         media_urls=["/cache/img_1.png"], media_types=["image/png"])
    recorder = _VisionRecorder()
    with patch("tools.vision_tools.vision_analyze_tool", recorder):
        out = asyncio.run(runner._prepare_inbound_message_text(
            event=event, source=source, history=list(history), session_key="tg:1"))
    return out, recorder


class _ChainWalkRecorder(_VisionRecorder):
    """The vision call's primary fails, so its auxiliary walk reaches a hosted chain entry."""

    def __init__(self):
        super().__init__()
        self.refused = []

    async def __call__(self, image_url, user_prompt=None, **kw):
        self.refused.append(route_hooks.auxiliary_refuses(HOSTED, task="vision", failed_provider="openrouter"))
        return await super().__call__(image_url, user_prompt, **kw)


def _prepare_with(runner, text, recorder):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="private", user_name="u")
    event = MessageEvent(text=text, message_type=MessageType.PHOTO, source=source,
                         media_urls=["/cache/img_1.png"], media_types=["image/png"])
    with patch("tools.vision_tools.vision_analyze_tool", recorder):
        return asyncio.run(runner._prepare_inbound_message_text(event=event, source=source, history=[],
                                                                session_key="tg:1"))


class TestInboundLabels:
    def test_gateway_vision_sees_labels_from_the_inbound_message(self, hooks):
        gate_calls = []
        _content_labeller(hooks, gate_calls)
        record = route_hooks.resolution_record(HOSTED, {"provider": "openrouter"})
        _text, recorder = _prepare_text(_vision_runner(record, session_id="inbound"), "acme invoice, what is this")
        assert recorder.calls == []
        assert gate_calls == [("auxiliary", ["acme:client"])]
        assert route_hooks.session_labels("inbound") == {"acme:client"}

    def test_a_failing_labeller_fails_closed_before_the_image_leaves(self, hooks):
        gate_calls = []
        _content_labeller(hooks, gate_calls, raises=True)
        record = route_hooks.resolution_record(HOSTED, {"provider": "openrouter"})
        _text, recorder = _prepare_text(_vision_runner(record), "what is this")
        assert recorder.calls == []
        assert gate_calls == []  # labels unknown: refused without asking the gate

    def test_cli_vision_sees_labels_from_the_inbound_message(self, hooks, tmp_path):
        gate_calls = []
        _content_labeller(hooks, gate_calls)
        agent = _make_agent(session_id="cli-inbound")
        route_hooks.bind_resolution_entry(agent, route_hooks.resolution_record(HOSTED, {"provider": "openrouter"}))
        _result, recorder, _img = _cli_preprocess(_cli(agent), tmp_path, text="acme invoice, what is this")
        assert recorder.calls == []
        assert gate_calls == [("auxiliary", ["acme:client"])]

    def test_the_vision_fallback_walk_on_the_primary_sees_the_inbound_labels(self, hooks):
        gate_calls = []
        _content_labeller(hooks, gate_calls)
        walk = _ChainWalkRecorder()
        _prepare_with(_vision_runner(None, session_id="primary"), "acme invoice, what is this", walk)
        assert walk.refused == [True]  # the primary failed: the hosted chain entry is refused
        assert gate_calls == [("auxiliary", ["acme:client"])]

    def test_a_failing_labeller_refuses_the_vision_fallback_walk(self, hooks):
        gate_calls = []
        _content_labeller(hooks, gate_calls, raises=True)
        walk = _ChainWalkRecorder()
        _prepare_with(_vision_runner(None), "what is this", walk)
        assert walk.refused == [True]
        assert gate_calls == []
        assert route_hooks._INBOUND_LABELS_KNOWN.get() is True  # reset after the call

    def test_an_unlabelled_message_is_admitted(self, hooks):
        gate_calls = []
        _content_labeller(hooks, gate_calls)
        record = route_hooks.resolution_record(HOSTED, {"provider": "openrouter"})
        _text, recorder = _prepare_text(_vision_runner(record), "what is this")
        assert len(recorder.calls) == 1
        assert gate_calls == [("auxiliary", [])]


# -- Sticker: Telegram sticker vision has no session route ---------------------------------------


def _sticker_text(tmp_path):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._accept_update = lambda: None
    file_obj = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"RIFFwebp")))
    sticker = SimpleNamespace(emoji="+", set_name="pack", is_animated=False, is_video=False,
                              file_unique_id="sticker-1", get_file=AsyncMock(return_value=file_obj))
    event = SimpleNamespace(text="")
    recorder = _VisionRecorder()
    with patch("tools.vision_tools.vision_analyze_tool", recorder), \
         patch("gateway.sticker_cache.get_cached_description", return_value=None), \
         patch("gateway.sticker_cache.cache_sticker_description_async", AsyncMock()), \
         patch("plugins.platforms.telegram.adapter.cache_image_from_bytes_async",
               AsyncMock(return_value=str(tmp_path / "s.webp"))):
        asyncio.run(adapter._handle_sticker(SimpleNamespace(sticker=sticker), event))
    return event.text, recorder


class TestStickerVision:
    def test_with_a_route_hook_loaded_no_sticker_image_leaves(self, hooks, tmp_path):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        text, recorder = _sticker_text(tmp_path)
        assert recorder.calls == []
        assert "a sticker with emoji +" in text

    def test_with_no_hook_the_sticker_is_described_as_before(self, tmp_path):
        text, recorder = _sticker_text(tmp_path)
        assert len(recorder.calls) == 1
        assert "a chart" in text


# -- Carry: the resident agent's held copies reach the rotated child -----------------------------


class TestRotatingCarryHeldCopies:
    def test_held_labels_survive_registry_eviction(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        resident = _make_agent(session_id="evicted-parent")
        resident._route_labels = ("evicted-parent", frozenset({"acme:client"}))
        resident._route_data_use = ("evicted-parent", frozenset({"acme:client-raw"}))
        assert route_hooks.session_labels("evicted-parent") == frozenset()  # the registry row is gone
        _write_rotating_compress("evicted-parent", "evicted-child", resident=resident)
        assert route_hooks.session_labels("evicted-child") == {"acme:client"}
        assert "acme:client-raw" in route_hooks.session_data_use("evicted-child")

    def test_a_resident_of_another_session_carries_nothing(self, hooks):
        hooks["route_custody"] = [lambda **_kw: {"action": "allow"}]
        resident = _make_agent(session_id="other")
        resident._route_labels = ("other", frozenset({"acme:client"}))
        _write_rotating_compress("plain-parent", "plain-child", resident=resident)
        assert route_hooks.session_labels("plain-child") == frozenset()
