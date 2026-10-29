"""An inline ``__HERMES_MOA_TURN_V1__`` marker applies a MoA preset only from a trusted caller or a
local session with ``moa.inline_marker`` on.

The marker carries its own preset (advisor models and aggregator). Decoded from message text, it
would let any sender pick the providers a turn fans out to. The API server and gateway platforms
such as Discord keep it as plain user text: the turn makes one main-model chat and no advisor call.

Real imports against a temp ``HERMES_HOME``, one loopback OpenAI-compatible endpoint that records
the model of every chat, and synthetic keys only.
"""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest
import hermes_yaml as yaml

from hermes_cli.moa_config import MOA_MARKER_PREFIX

MAIN_MODEL = "main-chat-model"
ADVISOR_MODELS = ("advisor-a", "advisor-b")
AGGREGATOR_MODEL = "advisor-aggregator"
PROMPT = "compare the two riffs"


def _marker(prompt=PROMPT):
    payload = {"prompt": prompt, "config": {
        "reference_models": [{"provider": "custom:main", "model": m} for m in ADVISOR_MODELS],
        "aggregator": {"provider": "custom:main", "model": AGGREGATOR_MODEL},
    }}
    return MOA_MARKER_PREFIX + base64.urlsafe_b64encode(json.dumps(payload).encode()).decode("ascii")


def _serve():
    """Loopback chat endpoint: answers every chat (plain or streamed) and records each request."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/chat/completions"):
                self.send_error(404)
                return
            seen.append(payload)
            body = {
                "id": "chatcmpl-fake", "object": "chat.completion", "created": 1,
                "model": payload.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "loopback answered"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            content_type = "application/json"
            if payload.get("stream"):
                body["object"] = "chat.completion.chunk"
                body["choices"][0]["delta"] = body["choices"][0].pop("message")
                raw = f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return server, thread, seen


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp home whose main model and every MoA slot resolve to one recording loopback."""
    from agent.auxiliary_client import _reset_aux_unhealthy_cache, shutdown_cached_clients
    from agent import moa_loop

    shutdown_cached_clients()
    _reset_aux_unhealthy_cache()
    monkeypatch.setattr(moa_loop, "_runtime_cache", {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("FAKE_MAIN_KEY", "synthetic-main-key-not-real")
    server, thread, seen = _serve()
    url = f"http://127.0.0.1:{server.server_port}/v1"

    def write(**moa):
        config = {
            "model": {"provider": "custom:main", "default": MAIN_MODEL, "context_length": 128000},
            "providers": {"main": {"base_url": url, "key_env": "FAKE_MAIN_KEY",
                                   "transport": "chat_completions"}},
            "auxiliary": {"title_generation": {"enabled": False}},
            "moa": moa,
        }
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    write()
    try:
        yield {"url": url, "seen": seen, "write": write, "path": tmp_path}
    finally:
        shutdown_cached_clients()
        _reset_aux_unhealthy_cache()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _models(seen):
    return [request.get("model") for request in seen]


def _last_user_text(request):
    content = [m for m in request["messages"] if m.get("role") == "user"][-1]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def _agent(url, platform, **kwargs):
    from run_agent import AIAgent

    return AIAgent(
        model=MAIN_MODEL, provider="custom", base_url=url, api_key="synthetic-main-key-not-real",
        api_mode="chat_completions", platform=platform, quiet_mode=True, enabled_toolsets=[],
        skip_context_files=True, skip_memory=True, save_trajectories=False, max_iterations=2,
        **kwargs,
    )


def _assert_plain_text_turn(seen):
    """Exactly one main-model chat, no advisor or aggregator call, the marker sent as typed."""
    assert _models(seen) == [MAIN_MODEL]
    assert _last_user_text(seen[0]).startswith(MOA_MARKER_PREFIX)


def _assert_moa_turn(seen):
    """The decoded preset ran: both advisors and the aggregator, then the main model on the prompt."""
    models = _models(seen)
    assert set(ADVISOR_MODELS) | {AGGREGATOR_MODEL} <= set(models), models
    main = [request for request in seen if request.get("model") == MAIN_MODEL]
    assert len(main) == 1, models
    assert PROMPT in _last_user_text(main[0])
    assert MOA_MARKER_PREFIX not in _last_user_text(main[0])


# -- API server (pilot T11) -------------------------------------------------------------------


def _api_app(adapter):
    from aiohttp import web

    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("inline_marker", [None, False, True])
async def test_api_server_marker_is_plain_text(home, monkeypatch, caplog, inline_marker):
    """A marker posted to a public home's ``/v1/chat/completions`` runs one main-model chat and no
    advisor, whatever ``moa.inline_marker`` says: the API server is not a local session."""
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    if inline_marker is not None:
        home["write"](inline_marker=inline_marker)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {
        "provider": "custom", "api_key": "synthetic-main-key-not-real", "base_url": home["url"],
        "api_mode": "chat_completions"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: MAIN_MODEL)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))

    with caplog.at_level("WARNING", logger="agent.conversation_loop"):
        async with TestClient(TestServer(_api_app(adapter))) as client:
            resp = await client.post("/v1/chat/completions", json={
                "model": "hermes-agent", "messages": [{"role": "user", "content": _marker()}]})
            body = await resp.json()

    assert resp.status == 200, body
    assert body["choices"][0]["message"]["content"] == "loopback answered"
    _assert_plain_text_turn(home["seen"])
    assert any("Inline MoA marker on a api_server turn" in r.getMessage() for r in caplog.records)


# -- Gateway platforms ------------------------------------------------------------------------


def test_discord_shaped_marker_is_plain_text(home):
    """A Discord thread message, built with the agent kwargs the gateway passes for a Discord
    source (gateway/session_execution.py), runs one main-model chat and no advisor."""
    home["write"](inline_marker=True)
    agent = _agent(home["url"], "discord", chat_id="12345", chat_type="thread", user_id="user-1",
                   gateway_session_key="agent:main:discord:thread:12345")

    result = agent.run_conversation(_marker())

    assert result["final_response"] == "loopback answered"
    _assert_plain_text_turn(home["seen"])


# -- Local sessions and trusted callers ------------------------------------------------------


def test_local_session_marker_is_plain_text_by_default(home):
    agent = _agent(home["url"], "cli")

    agent.run_conversation(_marker())

    _assert_plain_text_turn(home["seen"])


@pytest.mark.parametrize("value", [True, "true"])
def test_local_session_decodes_marker_when_inline_marker_on(home, value):
    home["write"](inline_marker=value)
    agent = _agent(home["url"], "cli")

    result = agent.run_conversation(_marker())

    assert result["final_response"] == "loopback answered"
    _assert_moa_turn(home["seen"])


def test_trusted_caller_decodes_marker_on_any_platform(home):
    """``moa_marker_trusted`` is set by internal callers only; it never comes from message text."""
    agent = _agent(home["url"], "api_server")

    agent.run_conversation(_marker(), moa_marker_trusted=True)

    _assert_moa_turn(home["seen"])


def test_inline_marker_key_defaults_off_and_survives_normalize():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.moa_config import moa_inline_marker_enabled, normalize_moa_config

    assert DEFAULT_CONFIG["moa"]["inline_marker"] is False
    assert normalize_moa_config({})["inline_marker"] is False
    assert normalize_moa_config({"inline_marker": True})["inline_marker"] is True
    assert moa_inline_marker_enabled({"inline_marker": "off"}) is False
    assert moa_inline_marker_enabled({"inline_marker": "nonsense"}) is False
    assert moa_inline_marker_enabled(None) is False
