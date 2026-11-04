"""``post_api_response`` (F8): each receipt must pass cade ``parseRouteReceipt``."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.route_receipt import ROUTE_RECEIPT_HEADERS, ROUTE_RECEIPT_SCHEMA, build_route_receipt, fire_route_receipt, receipt_headers
from hermes_cli.plugins import VALID_HOOKS, PluginManager

UNIFIED = "anthropic-ratelimit-unified-"
# xoxd-ai/cade src/quota-watch.ts, read at main 9e8937b.
RECEIPT_KEYS = {"schema", "home", "provider", "status", "headers", "usage", "at"}
INSTANT = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{1,9})?(Z|[+-]\d\d:\d\d)$")
HOME = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
QUOTA_WATCH_ALLOWLIST = {
    *(UNIFIED + n for n in ("status", "reset", "representative-claim", "5h-status", "5h-utilization", "5h-reset",
                            "7d-status", "7d-utilization", "7d-reset", "fallback-percentage", "overage-status")),
    "retry-after", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
}


def assert_parses(receipt):
    """The checks ``parseRouteReceipt`` applies, minus its provider enum: a plugin maps that."""
    assert set(receipt) == RECEIPT_KEYS
    assert receipt["schema"] == ROUTE_RECEIPT_SCHEMA == "cade.route-receipt/v0"
    assert HOME.match(receipt["home"])
    assert isinstance(receipt["status"], int) and 100 <= receipt["status"] <= 599
    assert len(receipt["headers"]) <= len(QUOTA_WATCH_ALLOWLIST)
    for name, value in receipt["headers"].items():
        assert name in QUOTA_WATCH_ALLOWLIST
        assert isinstance(value, str) and len(value) <= 64 and re.fullmatch(r"[A-Za-z0-9_.:+-]*", value)
    assert set(receipt["usage"]) == {"input", "output", "cacheRead", "cacheWrite"}
    assert all(isinstance(v, int) and v >= 0 for v in receipt["usage"].values())
    assert INSTANT.match(receipt["at"])


@pytest.fixture
def receipts(monkeypatch):
    manager = PluginManager()
    manager._discovered = True
    seen = []
    manager._hooks["post_api_response"] = [lambda **kw: seen.append(kw)]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    return seen


def _agent(**fields):
    agent = MagicMock()
    agent.provider, agent.session_id, agent.platform, agent.model = "anthropic", "s", "cli", "m"
    agent._last_route_headers = None
    agent._usage_summary_for_api_request_hook.return_value = {
        "input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 3, "cache_write_tokens": 4}
    for name, value in fields.items():
        setattr(agent, name, value)
    return agent


def test_the_allowlist_equals_the_quota_watch_allowlist():
    assert ROUTE_RECEIPT_HEADERS == QUOTA_WATCH_ALLOWLIST


def test_the_hook_is_registered():
    assert "post_api_response" in VALID_HOOKS


def test_headers_keep_the_allowlist_only():
    kept = receipt_headers({
        "Authorization": "Bearer sk-synthetic", "x-api-key": "sk-synthetic", "set-cookie": "a=b", "content-type": "x",
        "Retry-After": "60", UNIFIED + "Status": "rejected",
    })
    assert kept == {"retry-after": "60", UNIFIED + "status": "rejected"}


@pytest.mark.parametrize("value", ["a b", "x" * 65, "1;drop"], ids=["space", "long", "semicolon"])
def test_a_value_the_parser_would_refuse_is_dropped_and_the_receipt_survives(value):
    receipt = build_route_receipt(provider="anthropic", status=429, usage=None, headers={"retry-after": value, UNIFIED + "status": "rejected"})
    assert receipt["headers"] == {UNIFIED + "status": "rejected"}
    assert_parses(receipt)


def test_non_mapping_headers_keep_nothing():
    assert receipt_headers(["retry-after"]) == {}
    assert receipt_headers(None) == {}


def test_a_success_uses_the_captured_headers_and_usage(receipts):
    agent = _agent(_last_route_headers={UNIFIED + "status": "allowed"})
    fire_route_receipt(agent, status=200, task_id="t", turn_id="u", response=object())
    assert len(receipts) == 1
    receipt = receipts[0]["receipt"]
    assert_parses(receipt)
    assert receipt["status"] == 200
    assert receipt["headers"] == {UNIFIED + "status": "allowed"}
    assert receipt["usage"] == {"input": 10, "output": 2, "cacheRead": 3, "cacheWrite": 4}
    assert (receipts[0]["task_id"], receipts[0]["turn_id"], receipts[0]["session_id"]) == ("t", "u", "s")


def test_a_failed_attempt_never_reports_a_stale_capture(receipts):
    agent = _agent(_last_route_headers={UNIFIED + "status": "allowed"})
    fire_route_receipt(agent, status=429, task_id="t", turn_id="u", headers={"retry-after": "60"})
    assert receipts[0]["receipt"]["headers"] == {"retry-after": "60"}
    fire_route_receipt(agent, status=200, task_id="t", turn_id="u")
    assert receipts[1]["receipt"]["headers"] == {}


@pytest.mark.parametrize("status", [None, 0, 99, 600, True, "429"])
def test_an_attempt_with_no_http_status_has_no_receipt(receipts, status):
    fire_route_receipt(_agent(), status=status, task_id="t", turn_id="u")
    assert receipts == []


def test_a_hook_failure_never_reaches_the_turn(monkeypatch):
    manager = PluginManager()
    manager._discovered = True

    def boom(**kw):
        raise RuntimeError("synthetic")

    manager._hooks["post_api_response"] = [boom]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    fire_route_receipt(_agent(), status=200, task_id="t", turn_id="u")


def test_a_dispatch_failure_never_reaches_the_turn():
    with patch("hermes_cli.lifecycle.invoke_hook", side_effect=RuntimeError("synthetic")), \
            patch("hermes_cli.lifecycle.has_hook", return_value=True):
        fire_route_receipt(_agent(), status=200, task_id="t", turn_id="u")


def test_no_hook_loaded_builds_nothing():
    agent = _agent()
    with patch("hermes_cli.lifecycle.has_hook", return_value=False), patch("agent.route_receipt.build_route_receipt") as build:
        fire_route_receipt(agent, status=200, task_id="t", turn_id="u", response=object())
    build.assert_not_called()
    agent._usage_summary_for_api_request_hook.assert_not_called()


def test_a_message_never_rides_a_receipt():
    receipt = build_route_receipt(provider="anthropic", status=429, usage=None, headers=SimpleNamespace(items=lambda: [("x", "secret body")]))
    assert "secret" not in str(receipt)


def test_the_streaming_capture_keeps_the_allowlist_for_the_receipt():
    from agent.rate_limit_credits import RateLimitCreditsMixin

    agent = SimpleNamespace(provider="anthropic", _rate_limit_state=None)
    RateLimitCreditsMixin._capture_rate_limits(
        agent, SimpleNamespace(headers={"authorization": "Bearer sk-synthetic", UNIFIED + "status": "allowed"}))
    assert agent._last_route_headers == {UNIFIED + "status": "allowed"}
