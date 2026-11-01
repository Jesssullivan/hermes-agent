"""The shared route cooldown store (F9) and the classifier inputs that arm it (F7 and F8)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import extract_api_error_context
from agent.error_classifier import FailoverReason, classify_api_error
from agent.fallback_cooldown import _arm_rate_limit_cooldown
from agent.route_cooldown import (
    MAX_COOLDOWN_COUNT, MAX_COOLDOWN_SECONDS, _path, clear_route_cooldown, read_route_cooldown, record_route_cooldown,
    route_cooldown_remaining,
)
from agent.turn_recovery import _CodexSoftFailure, _codex_soft_failure_error

ROUTE = ("kimi-coding", "k3.1-pro")


def _arm(*route, seconds=3600.0, count=1):
    route = route or ROUTE
    record_route_cooldown(*route[:2], route[2] if len(route) > 2 else "", seconds=seconds, count=count, reason="rate_limit")


def _expire(*route):
    """Move the route's deadline into the past, keeping its count, instead of waiting for it."""
    path = Path(_path(*route, ""))
    path.write_text(json.dumps({**json.loads(path.read_text()), "until": time.time() - 1}))


@pytest.fixture(autouse=True)
def _synthetic_model_ids():
    with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
        yield


class TestStore:
    def test_a_recorded_deadline_is_read_back(self):
        _arm(seconds=3600, count=2)
        held = read_route_cooldown(*ROUTE)
        assert 3590 < held.remaining() <= 3600
        assert (held.count, held.reason) == (2, "rate_limit")
        assert 3590 < route_cooldown_remaining(*ROUTE) <= 3600

    def test_an_unknown_route_has_no_cooldown(self):
        assert read_route_cooldown(*ROUTE) is None
        assert route_cooldown_remaining(*ROUTE) == 0.0

    def test_an_expired_record_stays_for_its_count_and_holds_nothing(self):
        _arm(seconds=0, count=3)
        assert read_route_cooldown(*ROUTE).count == 3
        assert route_cooldown_remaining(*ROUTE) == 0.0

    def test_a_later_deadline_from_another_session_stands(self):
        _arm(seconds=3600)
        _arm(seconds=60)
        assert route_cooldown_remaining(*ROUTE) > 3000

    def test_two_models_on_one_provider_are_separate(self):
        _arm("openrouter", "a/one")
        assert route_cooldown_remaining("openrouter", "a/two") == 0.0

    def test_a_custom_name_is_one_route_with_or_without_its_url(self):
        # The agent records its resolved URL. A rendered ``custom:<name>`` chain entry has none.
        _arm("custom:cade-onprem", "local-model", "http://onprem.invalid/v1")
        assert route_cooldown_remaining("custom:cade-onprem", "local-model", None) > 0

    def test_two_custom_endpoints_are_separate_and_a_named_provider_ignores_the_url(self):
        _arm("custom", "m", "https://one.example/v1/")
        assert route_cooldown_remaining("custom", "m", "https://one.example/v1") > 0
        assert route_cooldown_remaining("custom", "m", "https://two.example/v1") == 0.0
        _arm("cade-siwc", "gpt-6.1-sol", "http://relay.invalid/v1")
        assert route_cooldown_remaining("cade-siwc", "gpt-6.1-sol", "") > 0

    def test_a_chain_entry_and_the_bound_model_land_on_one_file(self):
        with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m.replace(".", "-")):
            _arm("anthropic", "claude-opus-5.5")
            assert route_cooldown_remaining("anthropic", "claude-opus-5-5") > 0

    def test_the_file_holds_no_endpoint_or_credential(self):
        secret_url = "https://user:sk-synthetic@one.example/v1"
        _arm("custom", "m", secret_url)
        text = Path(_path("custom", "m", secret_url)).read_text()
        assert "sk-synthetic" not in text
        assert "one.example" not in text
        assert set(json.loads(text)) == {"provider", "model", "until", "count", "reason", "armed_at"}

    @pytest.mark.parametrize(
        "content",
        [
            "", "not json", "[]", '{"until": "soon"}', '{"until": 1e400, "count": 1, "reason": "r", "armed_at": 1}',
            '{"until": 1, "count": 1e400, "reason": "r", "armed_at": 1}',
        ],
        ids=["empty", "text", "list", "bad-value", "infinite", "infinite-count"],
    )
    def test_a_corrupt_file_is_no_cooldown(self, content):
        path = Path(_path(*ROUTE, ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        assert read_route_cooldown(*ROUTE) is None
        assert route_cooldown_remaining(*ROUTE) == 0.0

    def test_an_overflowing_count_can_be_written_over_and_cleared(self):
        path = Path(_path(*ROUTE, ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        bad = '{"until": 1, "count": 1e400, "reason": "r", "armed_at": 1}'
        path.write_text(bad)
        _arm(count=2)
        assert read_route_cooldown(*ROUTE).count == 2
        path.write_text(bad)
        clear_route_cooldown(*ROUTE)
        assert not path.exists()

    def test_a_huge_count_is_clamped(self):
        _arm(count=10 ** 9)
        assert read_route_cooldown(*ROUTE).count == MAX_COOLDOWN_COUNT

    def test_removing_the_file_that_names_the_route_lifts_the_hold(self):
        # The operator's remedy for a wrong hold: the file name is a hash, so the content finds it.
        _arm()
        _arm("anthropic", "claude-opus-5-5")
        store = Path(_path(*ROUTE, "")).parent
        named = [p for p in store.glob("route-*.json") if json.loads(p.read_text())["provider"] == ROUTE[0]]
        assert len(named) == 1
        named[0].unlink()
        assert route_cooldown_remaining(*ROUTE) == 0.0
        assert route_cooldown_remaining("anthropic", "claude-opus-5-5") > 0

    def test_a_deadline_past_the_cap_is_a_bad_file_and_not_a_hold(self):
        path = Path(_path(*ROUTE, ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"until": time.time() + 10 * MAX_COOLDOWN_SECONDS, "count": 1, "reason": "r", "armed_at": 1}))
        assert read_route_cooldown(*ROUTE) is None

    def test_a_record_longer_than_the_cap_is_clamped(self):
        _arm(seconds=100 * MAX_COOLDOWN_SECONDS)
        assert route_cooldown_remaining(*ROUTE) <= MAX_COOLDOWN_SECONDS

    def test_an_unwritable_home_fails_open(self, tmp_path):
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("x")
        with patch("hermes_constants.get_hermes_home", return_value=blocked):
            _arm()
            assert route_cooldown_remaining(*ROUTE) == 0.0
            clear_route_cooldown(*ROUTE)

    def test_a_served_response_clears_the_route(self):
        _arm()
        clear_route_cooldown(*ROUTE, started_at=time.time() + 1)
        assert read_route_cooldown(*ROUTE) is None
        assert not os.path.exists(_path(*ROUTE, ""))

    def test_a_stream_that_began_before_the_wall_does_not_clear_it(self):
        started = time.time() - 30
        _arm()
        clear_route_cooldown(*ROUTE, started_at=started)
        assert route_cooldown_remaining(*ROUTE) > 0

    def test_clearing_an_unknown_route_is_a_no_op(self):
        clear_route_cooldown(*ROUTE)


def _agent(**fields):
    base = dict(provider="anthropic", model="claude-opus-5-5", base_url="", _primary_runtime={"provider": "anthropic"})
    return SimpleNamespace(**{**base, **fields})


class TestArming:
    OPUS = ("anthropic", "claude-opus-5-5")

    def test_a_fresh_agent_continues_the_doubling_once_the_window_has_passed(self):
        for seconds in (60, 120, 240):
            assert _arm_rate_limit_cooldown(_agent(), FailoverReason.rate_limit) == seconds
            _expire(*self.OPUS)

    def test_a_burst_inside_one_window_counts_once(self):
        # Nine in-flight walls of one throttle. Per arm this reached count 9 and 14400 seconds.
        armed = [_arm_rate_limit_cooldown(_agent(), FailoverReason.rate_limit) for _ in range(9)]
        assert armed[0] == 60 and max(armed) == 60
        held = read_route_cooldown(*self.OPUS)
        assert held.count == 1
        assert 0 < held.remaining() <= 60

    def test_the_next_window_after_a_burst_is_the_next_level(self):
        for _ in range(9):
            _arm_rate_limit_cooldown(_agent(), FailoverReason.rate_limit)
        _expire(*self.OPUS)
        assert _arm_rate_limit_cooldown(_agent(), FailoverReason.rate_limit) == 120
        assert read_route_cooldown(*self.OPUS).count == 2

    def test_a_reused_agent_keeps_its_count_inside_the_window(self):
        agent = _agent()
        _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit)
        _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit)
        assert agent._rate_limit_backoff_count == 1

    def test_a_wall_inside_a_window_keeps_the_later_deadline(self):
        record_route_cooldown(*self.OPUS, "", seconds=3600, count=2, reason="rate_limit")
        agent = _agent()
        assert 3590 < _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit) <= 3600
        held = read_route_cooldown(*self.OPUS)
        assert held.count == 2
        assert 3590 < held.remaining() <= 3600
        assert 3590 < agent._rate_limited_until - time.monotonic() <= 3600

    def test_a_named_reset_inside_a_window_extends_it_and_keeps_the_count(self):
        record_route_cooldown(*self.OPUS, "", seconds=60, count=1, reason="rate_limit")
        armed = _arm_rate_limit_cooldown(_agent(), FailoverReason.rate_limit, reset_at=time.time() + 900)
        assert 890 < armed <= 900
        held = read_route_cooldown(*self.OPUS)
        assert held.count == 1
        assert 890 < held.remaining() <= 900

    def test_a_hop_wall_inside_a_window_keeps_the_count(self):
        agent = _agent(provider="kimi-coding", model="k3.1-pro", _fallback_activated=True)
        for _ in range(4):
            assert _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit) is None
        assert read_route_cooldown("kimi-coding", "k3.1-pro").count == 1

    def test_a_fallback_hop_records_the_route_and_leaves_the_primary_deadline(self):
        agent = _agent(provider="kimi-coding", model="k3.1-pro", _fallback_activated=True, _rate_limited_until=7.0,
                       _rate_limit_backoff_count=1)
        reset = time.time() + 900
        assert _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit, reset_at=reset) is None
        assert agent._rate_limited_until == 7.0
        assert agent._rate_limit_backoff_count == 1
        assert 880 < route_cooldown_remaining("kimi-coding", "k3.1-pro") <= 900

    def test_a_failure_that_is_not_a_rate_limit_records_nothing(self):
        assert _arm_rate_limit_cooldown(_agent(), FailoverReason.server_error) is None
        assert read_route_cooldown("anthropic", "claude-opus-5-5") is None


class TestSoftFailureCodes:
    """F7: a plan window that arrives as a bare code with no HTTP status."""

    @pytest.mark.parametrize("code", ["subscription_sharing_usage_limit_exceeded", "usage_limit_reached"])
    def test_a_bare_usage_code_is_a_rate_limit(self, code):
        classified = classify_api_error(_CodexSoftFailure({"code": code}), provider="cade-siwc", model="gpt-6.1-sol")
        assert classified.reason == FailoverReason.rate_limit
        assert classified.should_fallback is True

    def test_a_billing_code_stays_billing(self):
        classified = classify_api_error(_CodexSoftFailure({"code": "insufficient_quota"}), provider="cade-siwc", model="m")
        assert classified.reason == FailoverReason.billing

    def test_an_unrelated_code_stays_unknown(self):
        classified = classify_api_error(_CodexSoftFailure({"code": "unexplained_failure"}), provider="cade-siwc", model="m")
        assert classified.reason == FailoverReason.unknown

    @pytest.mark.parametrize(
        "error",
        [{"code": "c", "resets_in_seconds": 900}, SimpleNamespace(code="c", message=None, resets_in_seconds=900)],
        ids=["dict", "sdk-object"],
    )
    def test_a_numeric_reset_field_survives_the_reshaping(self, error):
        fields = _codex_soft_failure_error(SimpleNamespace(error=error))
        assert fields == {"code": "c", "resets_in_seconds": 900}
        context = extract_api_error_context(_CodexSoftFailure(fields))
        assert 880 < context["reset_at"] - time.time() <= 900

    def test_only_reset_fields_keep_a_number(self):
        fields = _codex_soft_failure_error(SimpleNamespace(error={"code": "c", "param": 3, "resets_at": True}))
        assert fields == {"code": "c"}


class _Wall(Exception):
    def __init__(self, headers):
        super().__init__("Error code: 429")
        self.status_code = 429
        self.body = {}
        self.response = SimpleNamespace(status_code=429, headers=headers)


UNIFIED = "anthropic-ratelimit-unified-"


class TestUnifiedReset:
    """F8: the reset on an Anthropic plan-window refusal. Shape U: these names are not measured live."""

    def test_the_overall_reset_of_a_refusal_is_an_instant_and_not_a_duration(self):
        reset = int(time.time()) + 3600
        context = extract_api_error_context(_Wall({UNIFIED + "status": "rejected", UNIFIED + "reset": str(reset)}))
        assert context["reset_at"] == reset

    def test_an_iso_reset_parses(self):
        context = extract_api_error_context(
            _Wall({UNIFIED + "status": "limited", UNIFIED + "reset": "2099-01-01T00:00:00+00:00"}))
        assert context["reset_at"] > time.time()

    def test_a_refused_windows_own_reset_counts(self):
        reset = int(time.time()) + 7200
        context = extract_api_error_context(_Wall({UNIFIED + "7d-status": "rejected", UNIFIED + "7d-reset": str(reset)}))
        assert context["reset_at"] == reset

    def test_a_window_that_did_not_refuse_gives_no_reset(self):
        headers = {UNIFIED + "status": "allowed", UNIFIED + "reset": str(int(time.time()) + 3600),
                   UNIFIED + "5h-status": "allowed", UNIFIED + "5h-reset": str(int(time.time()) + 60)}
        assert "reset_at" not in extract_api_error_context(_Wall(headers))

    def test_a_past_reset_gives_no_reset(self):
        headers = {UNIFIED + "status": "rejected", UNIFIED + "reset": str(int(time.time()) - 60)}
        assert "reset_at" not in extract_api_error_context(_Wall(headers))

    def test_a_refused_windows_reset_beats_a_short_retry_after(self):
        # Shape U. A closed plan window with a short Retry-After would otherwise wall per minute.
        reset = int(time.time()) + 3600
        headers = {"retry-after": "90", UNIFIED + "status": "rejected", UNIFIED + "reset": str(reset)}
        assert extract_api_error_context(_Wall(headers))["reset_at"] == reset

    def test_retry_after_stands_when_no_window_refused(self):
        headers = {"retry-after": "90", UNIFIED + "status": "allowed", UNIFIED + "reset": str(int(time.time()) + 3600)}
        assert 80 < extract_api_error_context(_Wall(headers))["reset_at"] - time.time() <= 90

    def test_retry_after_stands_when_the_refused_reset_has_passed(self):
        headers = {"retry-after": "90", UNIFIED + "status": "rejected", UNIFIED + "reset": str(int(time.time()) - 60)}
        assert 80 < extract_api_error_context(_Wall(headers))["reset_at"] - time.time() <= 90

    def test_a_usage_limit_refusal_with_a_unified_reset_is_a_rate_limit_and_not_billing(self):
        class Refused(_Wall):
            def __str__(self):
                return "usage limit reached"

        headers = {UNIFIED + "status": "rejected", UNIFIED + "reset": str(int(time.time()) + 3600)}
        error = Refused(headers)
        error.body = {"error": {"message": "usage limit reached"}}
        assert classify_api_error(error, provider="anthropic", model="m").reason == FailoverReason.rate_limit
