"""``auxiliary.main_model_fallback`` and ``compression.main_model_fallback`` keep side traffic off
the main model.

An explicit-provider auxiliary slot whose endpoint cannot serve (here: a closed port) retries on
the main agent model once its own ``fallback_chain`` is exhausted, and a compression summary that
fails on a distinct compression model retries once on the main model. Both nets stay on by
default. With the keys false, the call fails closed and the main model receives no request.

Real imports against a temp ``HERMES_HOME``, loopback fakes and synthetic keys only.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading

import pytest
import hermes_yaml as yaml

MAIN_MODEL = "main-chat-model"
ONPREM_MODEL = "onprem-aux-model"

# Every text slot the auxiliary ladder serves except compression (probed on its own below).
AUX_TEXT_SLOTS = (
    "skills_hub", "approval", "mcp", "title_generation", "memory_query_rewrite", "tts_audio_tags",
    "voice_chat", "triage_specifier", "kanban_decomposer", "profile_describer", "goal_judge",
    "curator", "monitor", "background_review",
)


def _serve(status):
    """Loopback OpenAI-compatible endpoint answering every chat with ``status``; records models."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/chat/completions"):
                self.send_error(404)
                return
            seen.append(payload.get("model"))
            if status == 200:
                body = {
                    "id": "chatcmpl-fake", "object": "chat.completion", "created": 1,
                    "model": payload.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "main model answered"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            else:
                body = {"error": {"message": "switch busy: voice entry holds the GPU",
                                  "type": "conflict", "code": "conflict"}}
            content_type = "application/json"
            if status == 200 and payload.get("stream"):
                body["object"] = "chat.completion.chunk"
                body["choices"][0]["delta"] = body["choices"][0].pop("message")
                raw = f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                raw = json.dumps(body).encode()
            self.send_response(status)
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


def _closed_port_url():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


@pytest.fixture
def endpoints(monkeypatch):
    from agent.auxiliary_client import _reset_aux_unhealthy_cache, shutdown_cached_clients

    shutdown_cached_clients()
    _reset_aux_unhealthy_cache()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("FAKE_MAIN_KEY", "synthetic-main-key-not-real")
    monkeypatch.setenv("FAKE_ONPREM_KEY", "synthetic-onprem-key-not-real")
    servers = {"main": _serve(200), "conflict": _serve(409)}
    try:
        yield {name: (f"http://127.0.0.1:{srv.server_port}/v1", seen)
               for name, (srv, _thread, seen) in servers.items()}
    finally:
        shutdown_cached_clients()
        _reset_aux_unhealthy_cache()
        for srv, thread, _seen in servers.values():
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=5)


def _write_config(home, main_url, onprem_url, *, aux_key=None, compression_key=None):
    slot = {"provider": "custom:onprem", "model": ONPREM_MODEL, "timeout": 5}
    config = {
        "model": {"provider": "custom:main", "default": MAIN_MODEL, "context_length": 128000},
        "providers": {
            "main": {"base_url": main_url, "key_env": "FAKE_MAIN_KEY", "transport": "chat_completions"},
            "onprem": {"base_url": onprem_url, "key_env": "FAKE_ONPREM_KEY", "transport": "chat_completions"},
        },
        # Capacity errors fail fast: the probe counts requests, not retry timing.
        "auxiliary": {"transient_retries": 0, "compression": dict(slot),
                      **{task: dict(slot) for task in AUX_TEXT_SLOTS}},
        "compression": {},
    }
    if aux_key is not None:
        config["auxiliary"]["main_model_fallback"] = aux_key
    if compression_key is not None:
        config["compression"]["main_model_fallback"] = compression_key
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _call_every_slot():
    from agent.auxiliary_client import call_llm

    outcomes = {}
    for task in AUX_TEXT_SLOTS:
        try:
            resp = call_llm(task=task, messages=[{"role": "user", "content": "side task"}], timeout=5)
            outcomes[task] = resp.choices[0].message.content
        except Exception as exc:  # the probe records how each slot ended
            outcomes[task] = exc
    return outcomes


def test_closed_onprem_slots_retry_on_main_model_by_default(tmp_path, monkeypatch, endpoints):
    """Upstream default kept: each of the 14 slots answers through the main model."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    main_url, main_seen = endpoints["main"]
    _write_config(tmp_path, main_url, _closed_port_url())

    outcomes = _call_every_slot()

    assert outcomes == {task: "main model answered" for task in AUX_TEXT_SLOTS}
    assert main_seen == [MAIN_MODEL] * len(AUX_TEXT_SLOTS)


@pytest.mark.parametrize("value", [False, "false"])
def test_closed_onprem_slots_fail_closed_when_main_model_fallback_off(
    tmp_path, monkeypatch, endpoints, caplog, value,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    main_url, main_seen = endpoints["main"]
    _write_config(tmp_path, main_url, _closed_port_url(), aux_key=value)

    with caplog.at_level("WARNING"):
        outcomes = _call_every_slot()

    assert main_seen == []
    assert all(isinstance(result, Exception) for result in outcomes.values()), outcomes
    refusals = [r for r in caplog.records if "auxiliary.main_model_fallback: false" in r.getMessage()]
    assert {task for task in AUX_TEXT_SLOTS if any(f"Auxiliary {task}:" in r.getMessage() for r in refusals)} \
        == set(AUX_TEXT_SLOTS)


def test_session_scoped_auxiliary_section_turns_the_net_off(tmp_path, monkeypatch, endpoints):
    """A gateway session's frozen ``auxiliary:`` section decides, not the file on disk."""
    from agent.auxiliary_task_config import session_auxiliary_scope
    from hermes_cli.config import load_config_readonly

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    main_url, main_seen = endpoints["main"]
    _write_config(tmp_path, main_url, _closed_port_url())
    frozen = dict(load_config_readonly()["auxiliary"], main_model_fallback=False)

    with session_auxiliary_scope(frozen):
        outcomes = _call_every_slot()

    assert main_seen == []
    assert all(isinstance(result, Exception) for result in outcomes.values()), outcomes


def _compressor(main_url):
    """The compressor a real agent builds (agent_init), so the config key is read as in production."""
    from run_agent import AIAgent

    agent = AIAgent(
        provider="custom:main", base_url=main_url, api_key="synthetic-main-key-not-real",
        model=MAIN_MODEL, enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False,
    )
    return agent.context_compressor


_TURNS = [
    {"role": "user", "content": "Plan the release."},
    {"role": "assistant", "content": "Drafted the release plan and the checklist."},
    {"role": "user", "content": "Ship the notes."},
    {"role": "assistant", "content": "Notes shipped."},
]


@pytest.mark.parametrize("aux_key", [None, False])
def test_compression_409_retries_on_main_model_by_default(tmp_path, monkeypatch, endpoints, aux_key):
    """Upstream default kept; the auxiliary key alone does not stop the compressor's own retry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    main_url, main_seen = endpoints["main"]
    onprem_url, onprem_seen = endpoints["conflict"]
    _write_config(tmp_path, main_url, onprem_url, aux_key=aux_key)
    compressor = _compressor(main_url)
    assert compressor.main_model_fallback is True

    summary = compressor._generate_summary(list(_TURNS))

    assert summary is not None and "main model answered" in summary
    assert onprem_seen and set(onprem_seen) == {ONPREM_MODEL}
    assert main_seen == [MAIN_MODEL]


@pytest.mark.parametrize("value", [False, "false"])
def test_compression_409_fails_closed_when_main_model_fallback_off(tmp_path, monkeypatch, endpoints, value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    main_url, main_seen = endpoints["main"]
    onprem_url, onprem_seen = endpoints["conflict"]
    _write_config(tmp_path, main_url, onprem_url, compression_key=value)
    compressor = _compressor(main_url)
    assert compressor.main_model_fallback is False

    assert compressor._generate_summary(list(_TURNS)) is None

    assert onprem_seen and set(onprem_seen) == {ONPREM_MODEL}
    assert main_seen == []
    assert "compression.main_model_fallback is false" in (compressor._last_summary_error or "")
    assert not getattr(compressor, "_summary_model_fallen_back", False)


def test_closed_onprem_compression_needs_both_keys_off(tmp_path, monkeypatch, endpoints):
    """A connection error walks the auxiliary ladder (main-agent net) and then the compressor's retry:
    with both keys false neither reaches the main model."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    main_url, main_seen = endpoints["main"]
    _write_config(tmp_path, main_url, _closed_port_url(), aux_key=False, compression_key=False)
    compressor = _compressor(main_url)

    assert compressor._generate_summary(list(_TURNS)) is None

    assert main_seen == []
