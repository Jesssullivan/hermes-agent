"""TIN-7696 D17: api.kimi.ai takes the same gates as api.kimi.com. No test leaves the process."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.anthropic_adapter import _auth_style, build_anthropic_client
from agent.agent_init import _host_default_headers_factory
from agent.auxiliary_client import _endpoint_default_headers, _endpoint_speaks_anthropic_messages, _to_openai_base_url
from agent.client_lifecycle import _ROUTE_DEFAULT_HEADERS
from agent.message_sanitization import needs_reasoning_echo, reasoning_echo_family
from agent.model_metadata import _endpoint_scoped_context_length
from hermes_cli.providers import host_mandated_api_mode
from hermes_cli.runtime_provider import _detect_api_mode_for_url
from run_agent import AIAgent
from trajectory_compressor import TrajectoryCompressor
from utils import KIMI_CODE_HOSTS, KIMI_CODE_USER_AGENT

BASES = [f"https://{host}/coding{suffix}" for host in KIMI_CODE_HOSTS for suffix in ("", "/v1")]
PLUGIN = Path(__file__).resolve().parents[2] / "plugins/model-providers/kimi-coding/__init__.py"


def test_both_hosts_and_the_model_user_agent_stay_pinned():
    assert KIMI_CODE_HOSTS == ("api.kimi.com", "api.kimi.ai")
    assert KIMI_CODE_USER_AGENT == "claude-code/0.1.0"


@pytest.mark.parametrize("base", BASES)
def test_transport_gates_pick_anthropic_messages(base):
    assert _detect_api_mode_for_url(base) == "anthropic_messages"
    assert host_mandated_api_mode(base) == "anthropic_messages"
    assert _endpoint_speaks_anthropic_messages(base)


@pytest.mark.parametrize("base", BASES)
def test_openai_wire_clients_send_the_model_user_agent(base):
    factory = _host_default_headers_factory(base)
    assert factory is not None and factory("key", base) == {"User-Agent": KIMI_CODE_USER_AGENT}
    assert _endpoint_default_headers(base, "kimi-coding")["User-Agent"] == KIMI_CODE_USER_AGENT


@pytest.mark.parametrize("host", KIMI_CODE_HOSTS)
def test_rebuilt_openai_clients_and_the_coding_rewrite_cover_each_host(host):
    assert host in [row[0] for row in _ROUTE_DEFAULT_HEADERS]
    assert _to_openai_base_url(f"https://{host}/coding") == f"https://{host}/coding/v1"
    assert _endpoint_scoped_context_length("k3", f"https://{host}/coding") == 1_048_576


@pytest.mark.parametrize("base", BASES)
@patch("agent.process_bootstrap.OpenAI")
def test_a_rebuilt_openai_client_sends_the_model_user_agent(mock_openai, base):
    """/model and credential rotation rebuild client kwargs through this path. Attribution stays."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._apply_client_headers_for_base_url(base)
    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"] == KIMI_CODE_USER_AGENT
    assert headers["X-Title"] == "Hermes Agent"


@pytest.mark.parametrize("base", BASES)
def test_the_reasoning_echo_rule_covers_each_host(base):
    """A custom provider name reaches the rule through the host alone. Tool-call turns 400 without the pad."""
    assert reasoning_echo_family("custom", "other-model", base) == "kimi"
    assert needs_reasoning_echo("custom", "other-model", base)


@pytest.mark.parametrize("base", BASES)
def test_the_trajectory_compressor_detects_each_host(base):
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = SimpleNamespace(base_url=base)
    assert compressor._detect_provider() == "kimi-coding"


@pytest.mark.parametrize("base", BASES)
def test_the_wire_request_on_each_base(base):
    """/coding and /coding/v1 reach the same path: the client strips /v1 and the SDK adds /v1/messages."""
    assert _auth_style("oauth-access-token", base, base) == "kimi"
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant", "model": "k3.1-pro",
                                         "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = build_anthropic_client("oauth-access-token", base)
    wired = client.with_options(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    wired.messages.create(model="k3.1-pro", max_tokens=1, messages=[{"role": "user", "content": "hi"}])
    request = seen[0]
    assert str(request.url) == f"https://{httpx.URL(base).host}/coding/v1/messages"
    assert request.headers["x-api-key"] == "oauth-access-token"
    assert "authorization" not in request.headers
    assert request.headers["user-agent"] == KIMI_CODE_USER_AGENT


@pytest.mark.parametrize("base", BASES)
def test_model_discovery_confirms_each_host(base):
    spec = importlib.util.spec_from_file_location("kimi_coding_plugin_under_test", PLUGIN)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    assert plugin._is_confirmed_kimi_coding_url(base)


@pytest.mark.parametrize("base", ["https://api.kimi.ai.example/coding", "https://example.test/api.kimi.ai/coding",
                                  "https://kimi.ai/coding", "https://api.kimi.ai/other"])
def test_lookalikes_do_not_take_the_kimi_gates(base):
    assert _detect_api_mode_for_url(base) is None
    assert not _endpoint_speaks_anthropic_messages(base)
    assert _auth_style("token", base, base) != "kimi"
    assert reasoning_echo_family("custom", "other-model", base) is None
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = SimpleNamespace(base_url=base)
    assert compressor._detect_provider() == ""
