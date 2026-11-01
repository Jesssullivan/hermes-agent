"""Quota-wall failover harness for Cade's routes (TIN-7765, SPR-150 to SPR-153, SPR-301).

Two chains run through the real ``run_conversation`` loop with fake transports. These tests record
what this revision does at a quota wall.

The public-home chain is what the cade launcher renders at Anthropic stage ``active``: Opus 5.5 on
main turns with one Kimi Code fallback (xoxd-ai/cade ``scripts/hermes-profiles.py``
``public_model_blocks``).

Two more chains run the three routes SPR-301 names. The ruled one is Kimi, then Opus 5.5, then Sol
through the SIWC relay (J160(a), TIN-7765 ``bdb14417``). The other starts on Sol, the cade home's
live route today, then Opus, then Kimi, because that is where a wall lands until the render moves.
Main cade renders neither yet: ``interim_model_blocks`` sets ``fallback_providers: []``. The ruled
chain here is what draft xoxd-ai/cade#256 ``chain_model_blocks`` renders, so Sol carries
``route_custody: required`` and no base_url, and the warm on-prem tail is a ``custom:<name>`` entry
with no base_url. J160(b) keeps that tail, for C1 and C2 turns first, and TestRuledOrder pins the
turn falling through it while no local model answers. A required entry needs a plugin that answers
for it, so the ruled tests use the ``custody_gate`` fixture, a real plugin manager with a synthetic
gate, and TestRuledOrderWithNoGate pins the turn with none. Cade's own gate plugin does not exist
yet. These tests load no route hook unless they say so.

The error bodies and headers are synthetic. The live shapes of a Claude Max plan-window 429, a Kimi
Code 429 and a ChatGPT plan wall are not measured (xoxd-ai/cade ``docs/ROUTE_FAILOVER.md`` section
8). The relay's own answers are read from xoxd-ai/cade ``integrations/hermes-siwc/relay.mjs``. A
test that pins a shape says which part of the shape it relies on. No test calls a live model, opens
a socket or reads a file outside the per-test Hermes home that ``tests/conftest.py`` sets up.

``tests/agent/conftest.py`` makes ``jittered_backoff`` return 0.0, so these tests pin the number of
attempts and any wait a provider named. They do not measure jittered wait lengths.

Bounds. ``interruptible_backoff_sleep`` waits ``while time.time() < sleep_end`` in 200 ms slices,
and a provider ``Retry-After`` is honoured up to ``RETRY_AFTER_CAP_S`` (600 s) per retry, plus up to
120 s per auto-recovery cycle. A harness that only stubs ``time.sleep`` turns each of those waits
into a busy loop on the real clock. An earlier draft of this file did that and recorded every
200 ms slice in a list, so one run grew past 8 GB of memory. Here every backoff wait is replaced at
its call sites and recorded once, with the seconds Hermes asked for. Three more bounds make a
regression fail fast instead of spinning: an API-call budget, a ``time.sleep`` budget and a
per-test wall-clock alarm. Each raises ``HarnessBound``, a ``BaseException``, so no
``except Exception`` in the loop can swallow it.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.codex_runtime import _raise_stream_error
from agent.error_classifier import ClassifiedError, FailoverReason
from run_agent import AIAgent

SOL = ("cade-siwc", "gpt-6.1-sol")
OPUS = ("anthropic", "claude-opus-5-5")
KIMI = ("kimi-coding", "k3.1-pro")
# The public-home fallback block as cade renders it at stage active.
PUBLIC_FALLBACK = [{"provider": KIMI[0], "model": KIMI[1]}]
# The chain behind a Sol primary, the live route today. Cade renders no such block yet: the interim
# relay route has ``fallback_providers: []``.
THREE_ROUTE_CHAIN = [{"provider": OPUS[0], "model": OPUS[1]}, {"provider": KIMI[0], "model": KIMI[1]}]
# The relay is a loopback listener. No test connects, so a reserved name stands in for its address.
SIWC_BASE_URL = "http://siwc-relay.invalid/v1"
# J160(a): Kimi, then Opus, then Sol. The primary is Kimi, so these are the entries behind it. This
# is the shape of xoxd-ai/cade#256 ``chain_model_blocks``: the endpoint and wire of Sol and of the
# tail come from the ``providers:`` block in RENDERED_PROVIDERS, not from the entry.
RULED_CHAIN = [
    {"provider": OPUS[0], "model": OPUS[1]},
    {"provider": SOL[0], "model": SOL[1], "route_custody": "required"},
]
ONPREM_NAME = "cade-onprem"
ONPREM = ("custom:" + ONPREM_NAME, "local-model")
ONPREM_BASE_URL = "http://onprem.invalid/v1"
# J160(b): the on-prem tail stays. #256 renders it only while a local model is warm.
RULED_CHAIN_WITH_TAIL = [*RULED_CHAIN, {"provider": ONPREM[0], "model": ONPREM[1]}]
RENDERED_PROVIDERS = {
    SOL[0]: {"base_url": SIWC_BASE_URL, "api_mode": "codex_responses", "key_env": "SYNTHETIC_RELAY_KEY"},
    ONPREM_NAME: {"api": ONPREM_BASE_URL, "transport": "chat_completions", "key_env": "SYNTHETIC_ONPREM_KEY"},
}
# A plan window that reopens in four hours, as on 2026-10-29 when a Kimi plan hit its 5 h window.
FOUR_HOURS = 4 * 3600

# Harness bounds. A turn in these tests needs at most a handful of calls and waits.
MAX_API_CALLS = 16
MAX_BACKOFF_WAITS = 32
MAX_SLEEP_CALLS = 64
WALL_SECONDS = 60

# Every module that binds ``interruptible_backoff_sleep`` at import time, plus its home module for
# the call sites that import it inside a function (turn_response_check, turn_recovery_autorecover).
BACKOFF_SITES = (
    "agent.turn_recovery",
    "agent.turn_api_error",
    "agent.turn_empty_response",
)


REAL_SLEEP = time.sleep


@pytest.fixture(autouse=True)
def _synthetic_model_ids():
    """These model ids are synthetic, so the catalog must not rename them between a write to the
    shared store and a read of it. ``drive`` patches the same name for its turn."""
    with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
        yield


@pytest.fixture(autouse=True)
def _rendered_providers():
    """The ``providers:`` block #256 renders beside the chain, in the per-test Hermes home."""
    from hermes_constants import get_hermes_home

    (Path(get_hermes_home()) / "config.yaml").write_text(json.dumps({"providers": RENDERED_PROVIDERS}))


class HarnessBound(BaseException):
    """A harness bound was hit. BaseException, so the agent loop cannot catch it as an API error."""


@pytest.fixture(autouse=True)
def _wall_clock_bound():
    """Fail a test that runs longer than WALL_SECONDS instead of letting it spin."""
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _expired(signum, frame):
        raise HarnessBound(f"test ran past {WALL_SECONDS} s")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class ProviderError(Exception):
    """An SDK status error as the classifier reads it: status_code, body and response headers."""

    def __init__(self, status_code, message, body=None, headers=None):
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code
        self.body = body if body is not None else {"error": {"message": message}}
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


def opus_rate_limit(headers):
    """An Anthropic 429 envelope (type error, error.type rate_limit_error)."""
    message = "This request would exceed your account's rate limit. Please try again later."
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": message}}
    return ProviderError(429, message, body=body, headers=headers)


def unified_headers(reset_epoch):
    """The anthropic-ratelimit-unified-* names cade's quota watch reads (shape U)."""
    return {
        "anthropic-ratelimit-unified-status": "rejected",
        "anthropic-ratelimit-unified-5h-status": "rejected",
        "anthropic-ratelimit-unified-reset": str(int(reset_epoch)),
        "anthropic-ratelimit-unified-representative-claim": "five_hour",
    }


def kimi_rate_limit(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return ProviderError(429, "Too Many Requests: rate limit reached for this plan window", headers=headers)


def relay_rejected():
    """The relay's answer to any upstream refusal but a 401 (relay.mjs, the ``!upstream.ok`` branch).

    The upstream status, code, message and headers stop at the relay. A ChatGPT plan wall reaches
    Hermes in this shape, the same as an upstream 400 or 500, with no reset signal.
    """
    body = {"error": {"code": "siwc_upstream_rejected"}}
    return ProviderError(502, str(body), body=body)


def relay_reauthentication_required():
    """The relay's answer to an upstream 401."""
    body = {"error": {"code": "siwc_reauthentication_required"}}
    return ProviderError(401, str(body), body=body)


def relay_down():
    """What the SDK raises when the relay's listener refuses the connection."""
    return openai.APIConnectionError(request=httpx.Request("POST", f"{SIWC_BASE_URL}/responses"))


def relay_stream_broken():
    """The frame the relay writes when a stream breaks after its headers went out."""
    try:
        _raise_stream_error({"type": "error", "code": "siwc_stream_not_completed"})
    except Exception as frame_error:
        return frame_error


def responses_failed(code, **fields):
    """A ``response.failed`` terminal. The relay forwards this frame, so its code reaches Hermes."""
    return SimpleNamespace(
        output=[], output_text="", status="failed", incomplete_details=None, error={"code": code, **fields},
        usage=None, model="fake", id="resp_fake", service_tier=None,
    )


def forwarded_sol_wall(reset_in):
    """A 429 the relay does not send today: status, code and reset carried through.

    This is the shape ``_status_429`` needs. ``resets_in_seconds`` and ``retry-after`` are names this
    classifier reads. Whether the ChatGPT backend sends them is not measured.
    """
    message = "The usage limit has been reached"
    error = {"code": "usage_limit_reached", "message": message, "resets_in_seconds": reset_in}
    return ProviderError(429, message, body={"error": error}, headers={"retry-after": str(reset_in)})


def _text_response(text, api_mode):
    if api_mode == "codex_responses":
        part = SimpleNamespace(type="output_text", text=text)
        item = SimpleNamespace(type="message", role="assistant", status="completed", content=[part])
        return SimpleNamespace(
            output=[item], output_text=text, status="completed", incomplete_details=None, error=None,
            usage=None, model="fake", id="resp_fake", service_tier=None,
        )
    if api_mode == "anthropic_messages":
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
            model="fake", usage=SimpleNamespace(input_tokens=10, output_tokens=2), role="assistant", type="message",
        )
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="fake", usage=None)


def _fake_anthropic_client(*args, **kwargs):
    """No SDK client and no socket: every Opus call goes through the patched transport."""
    return MagicMock()


def _agent(route, fallback, **kwargs):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch("agent.anthropic_adapter.build_anthropic_client", _fake_anthropic_client),
        patch("tools.env_probe.warm_environment_probe_async", lambda: None),
        patch("tools.env_probe.get_environment_probe_line", lambda **kwargs: ""),
    ):
        agent = AIAgent(
            provider=route[0], model=route[1], quiet_mode=True, skip_context_files=True,
            skip_memory=True, fallback_model=fallback, **kwargs,
        )
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._api_max_retries = 3
    return agent


def _public_home_agent(fallback=PUBLIC_FALLBACK):
    return _agent(OPUS, fallback, api_key="sk-ant-oat-synthetic", base_url="https://api.anthropic.com")


def _sol_home_agent(fallback=THREE_ROUTE_CHAIN):
    """The cade home on the SIWC relay, as ``interim_model_blocks`` renders it, with a chain added."""
    return _agent(SOL, fallback, api_key="relay-bearer-synthetic", base_url=SIWC_BASE_URL, api_mode="codex_responses")


def _kimi_home_agent(fallback=RULED_CHAIN):
    """The ruled chain: Kimi primary, then Opus, then Sol."""
    return _agent(KIMI, fallback, api_key="kimi-synthetic", base_url="https://api.kimi.com/coding/")


ROUTE_BASE_URLS = {OPUS[0]: "https://api.anthropic.com", SOL[0]: SIWC_BASE_URL, ONPREM[0]: ONPREM_BASE_URL}


def _route_client(provider, *args, **kwargs):
    """A fake of ``resolve_provider_client``: the configured fallback client, and no socket."""
    client = MagicMock()
    client.api_key = "synthetic"
    client.base_url = kwargs.get("explicit_base_url") or ROUTE_BASE_URLS.get(provider, "https://api.kimi.com/coding/")
    client._custom_headers = None
    client.default_headers = None
    return client, kwargs.get("model")


class Gate:
    """A real plugin manager holding a synthetic gate: ``turn_route`` labels the turn, and
    ``route_custody`` answers through ``decide(candidate)`` and records each call in ``seen``."""

    LABELS = ["synthetic-label"]

    def __init__(self):
        from hermes_cli.plugins import PluginManager

        self.seen = []
        self.decide = lambda candidate: {"action": "allow"}
        self.manager = PluginManager()
        self.manager._discovered = True
        self.manager._hooks["turn_route"] = [lambda **kw: {"labels": self.LABELS}]
        self.manager._hooks["route_custody"] = [self._custody]

    def _custody(self, **kw):
        self.seen.append(kw)
        return self.decide(kw["candidate"])

    def judged(self, route):
        return [kw for kw in self.seen if (kw["candidate"]["provider"], kw["candidate"]["model"]) == route]


@pytest.fixture
def custody_gate():
    gate = Gate()
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=gate.manager):
        yield gate


class Run(SimpleNamespace):
    """One driven turn: result, calls as (provider, model) per API call, waits as (route, seconds)
    for each backoff Hermes asked for, sleeps as any direct main-thread ``time.sleep`` argument, and
    requests as the ``api_kwargs`` of each call."""


def drive(agent, script, max_api_calls=MAX_API_CALLS):
    """Run one turn. script(route, attempt) returns an exception to raise, the reply text for that
    call, or a whole response object."""
    calls, waits, sleeps, requests = [], [], [], []

    def fake_api_call(api_kwargs, *args, **kwargs):
        if len(calls) >= max_api_calls:
            raise HarnessBound(f"more than {max_api_calls} API calls: {calls[-4:]}")
        route = (agent.provider, agent.model)
        calls.append(route)
        requests.append(api_kwargs)
        outcome = script(route, sum(1 for c in calls if c == route))
        if isinstance(outcome, BaseException):
            raise outcome
        return _text_response(outcome, agent.api_mode) if isinstance(outcome, str) else outcome

    def fake_backoff(_agent, wait_time, _retry, **kwargs):
        # Returning None reports a completed wait with no interrupt.
        if len(waits) >= MAX_BACKOFF_WAITS:
            raise HarnessBound(f"more than {MAX_BACKOFF_WAITS} backoff waits")
        waits.append(((agent.provider, agent.model), float(wait_time)))

    def fake_sleep(seconds):
        # Only the turn's own thread is stubbed. Another thread (a subprocess wait, a pool worker)
        # keeps the real sleep, so it neither spins nor spends this turn's budget.
        if threading.current_thread() is not threading.main_thread():
            return REAL_SLEEP(seconds)
        if len(sleeps) >= MAX_SLEEP_CALLS:
            raise HarnessBound(f"more than {MAX_SLEEP_CALLS} time.sleep calls")
        sleeps.append(float(seconds))

    with ExitStack() as stack:
        for name in BACKOFF_SITES:
            module = __import__(name, fromlist=["interruptible_backoff_sleep"])
            stack.enter_context(patch.object(module, "interruptible_backoff_sleep", fake_backoff))
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call))
        stack.enter_context(patch.object(agent, "_persist_session"))
        stack.enter_context(patch.object(agent, "_save_trajectory"))
        stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        stack.enter_context(patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()))
        stack.enter_context(patch("agent.anthropic_adapter.build_anthropic_client", _fake_anthropic_client))
        stack.enter_context(patch("time.sleep", fake_sleep))
        stack.enter_context(patch("tools.env_probe.warm_environment_probe_async", lambda: None))
        stack.enter_context(patch("tools.env_probe.get_environment_probe_line", lambda **kwargs: ""))
        stack.enter_context(patch(
            "agent.auxiliary_client.resolve_provider_client", side_effect=_route_client))
        stack.enter_context(patch(
            "hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m))
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200000))
        result = agent.run_conversation("synthetic public-home turn")
    return Run(result=result, calls=calls, waits=waits, sleeps=sleeps, requests=requests)


def _reply(result):
    return str(result.get("final_response") or "")


def _waits_on(run, route):
    """Seconds of each backoff Hermes asked for while the turn was on ``route``."""
    return [seconds for on, seconds in run.waits if on == route]


def _opus_waits(run):
    return _waits_on(run, OPUS)


def _cooldown(agent):
    """Seconds left on the primary's cooldown. 0 when none is armed."""
    return max(0.0, (getattr(agent, "_rate_limited_until", 0) or 0) - time.monotonic())


def _ledger_file(route):
    from agent.route_cooldown import _path

    return Path(_path(route[0], route[1], ""))


def _ledger(route):
    """The route's shared cooldown record, or None."""
    from agent.route_cooldown import read_route_cooldown

    return read_route_cooldown(route[0], route[1], "")


def _shared_seconds(route):
    """Seconds left on the route's shared cooldown. 0 when none is armed."""
    held = _ledger(route)
    return held.remaining() if held is not None else 0.0


def _expire_cooldown(agent, route=None):
    """Move both deadlines into the past instead of waiting for them. The shared record stays, so
    its count still doubles the next cooldown. ``route`` defaults to the agent's primary."""
    primary = agent._primary_runtime
    route = route or (primary["provider"], primary["model"])
    agent._rate_limited_until = time.monotonic() - 1
    path = _ledger_file(route)
    if path.exists():
        record = json.loads(path.read_text())
        record["until"] = time.time() - 1
        path.write_text(json.dumps(record))


class TestHarnessBounds:
    """The bounds themselves: a runaway turn fails fast and records nothing unbounded."""

    def test_api_call_budget_stops_a_turn_that_never_ends(self):
        agent = _public_home_agent(fallback=None)
        agent._api_max_retries = MAX_API_CALLS * 4
        overloaded = ProviderError(503, "Service Unavailable")
        with pytest.raises(HarnessBound):
            drive(agent, lambda route, attempt: overloaded)

    def test_capped_retry_after_is_recorded_once_per_wait(self):
        # Retry-After far past the cap: Hermes asks for at most RETRY_AFTER_CAP_S per wait, and the
        # harness records the request once instead of 3,000 slices of 200 ms.
        from agent.retry_utils import RETRY_AFTER_CAP_S

        agent = _public_home_agent(fallback=None)
        wall = opus_rate_limit({"retry-after": str(FOUR_HOURS)})
        run = drive(agent, lambda route, attempt: wall)
        assert run.waits, run
        assert len(run.waits) <= MAX_BACKOFF_WAITS
        assert max(seconds for _, seconds in run.waits) <= RETRY_AFTER_CAP_S


class TestOpusWallMovesToKimi:
    """SPR-150 for a public home: a quota wall on Opus moves the turn to Kimi Code."""

    def test_rate_limit_with_retry_after_falls_back_without_waiting(self):
        agent = _public_home_agent()
        wall = opus_rate_limit({"retry-after": str(FOUR_HOURS), **unified_headers(time.time() + FOUR_HOURS)})

        run = drive(agent, lambda route, attempt: wall if route == OPUS else "answered on kimi")
        assert run.calls == [OPUS, KIMI]
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on kimi"
        # The eager fallback moves before any backoff: the plan window is never waited out on Opus.
        assert _opus_waits(run) == []
        assert (agent.provider, agent.model) == KIMI

    def test_unified_headers_alone_still_fall_back(self):
        # Only the anthropic-ratelimit-unified-* names, no Retry-After. F8 reads the unified reset,
        # so the cooldown runs to the window's end. Those header names are shape U: not measured live.
        agent = _public_home_agent()
        wall = opus_rate_limit(unified_headers(time.time() + FOUR_HOURS))

        run = drive(agent, lambda route, attempt: wall if route == OPUS else "answered on kimi")
        assert run.calls == [OPUS, KIMI]
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on kimi"
        assert _opus_waits(run) == []
        assert FOUR_HOURS - 60 < _cooldown(agent) <= FOUR_HOURS

    def test_usage_limit_wording_without_reset_falls_back(self):
        # A wall phrased as a usage limit with no reset signal. Whichever reason it classifies as,
        # the turn reaches Kimi.
        agent = _public_home_agent()
        wall = ProviderError(429, "usage limit reached for this subscription")

        run = drive(agent, lambda route, attempt: wall if route == OPUS else "answered on kimi")
        assert run.calls[0] == OPUS and run.calls[-1] == KIMI
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on kimi"


class TestTransientStaysOnOpus:
    """A transient failure is not a quota wall (cade docs/ROUTE_FAILOVER.md section 2)."""

    def test_overloaded_529_retries_opus_before_any_switch(self):
        agent = _public_home_agent()
        overloaded = ProviderError(
            529, "Overloaded",
            body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )

        def script(route, attempt):
            if route == OPUS and attempt == 1:
                return overloaded
            return "answered on " + route[0]

        run = drive(agent, script)
        assert run.calls == [OPUS, OPUS]
        assert _reply(run.result) == "answered on anthropic"


class TestBothWalls:
    """SPR-151: with every route exhausted the turn must not end in silence."""

    def test_opus_and_kimi_walls_end_the_turn_with_text(self):
        agent = _public_home_agent()
        opus_wall = opus_rate_limit({"retry-after": str(FOUR_HOURS)})
        kimi_wall = kimi_rate_limit(retry_after=FOUR_HOURS)

        run = drive(agent, lambda route, attempt: opus_wall if route == OPUS else kimi_wall)
        assert run.calls[0] == OPUS and KIMI in run.calls
        assert run.result["completed"] is False
        # Hermes ends the turn with its own error text. It is not Cade's SPR-151 status reply:
        # that reply needs the F6 on_routes_exhausted hook (Jesssullivan/hermes-agent#3) or the
        # cade route gate in enforce mode.
        assert _reply(run.result).strip(), run.result
        assert _opus_waits(run) == []
        # Measured at this revision: the last route keeps its retries, so the turn waits out the
        # capped Retry-After on Kimi before the error text. Each wait is RETRY_AFTER_CAP_S, once per
        # retry after the first attempt. A public home at both walls answers that many minutes late.
        from agent.retry_utils import RETRY_AFTER_CAP_S

        assert _waits_on(run, KIMI) == [float(RETRY_AFTER_CAP_S)] * (agent._api_max_retries - 1)

    def test_no_fallback_chain_ends_the_turn_on_the_wall(self):
        # softproxy on Kimi renders no fallback (cade docs/HERMES_GATEWAY.md, softproxy routes).
        agent = _public_home_agent(fallback=None)
        wall = opus_rate_limit({"retry-after": str(FOUR_HOURS)})

        run = drive(agent, lambda route, attempt: wall)
        assert set(run.calls) == {OPUS}
        assert run.result["completed"] is False


class TestNextTurn:
    """What the turn after a wall does when the same agent runs it.

    ``_arm_rate_limit_cooldown`` (agent/fallback_cooldown.py) runs when a rate-limit or billing
    failure leaves the primary. It takes the provider's reset, and with none it uses 60 s doubling to
    4 h. ``restore_primary_runtime`` then keeps that agent on the fallback until the deadline passes.

    The same deadline is in the shared store (agent/route_cooldown.py), so an agent built after the
    wall holds it too. TestGatewayEviction pins that case.
    """

    @staticmethod
    def _script(wall):
        return lambda route, attempt: wall if route == OPUS else "answered on kimi"

    @pytest.mark.parametrize("reset_in", [90, FOUR_HOURS])
    def test_inside_the_cooldown_the_next_turn_starts_on_kimi(self, reset_in):
        agent = _public_home_agent()
        script = self._script(opus_rate_limit({"retry-after": str(reset_in)}))
        drive(agent, script)
        run = drive(agent, script)
        assert run.calls == [KIMI]
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on kimi"

    def test_after_the_cooldown_the_next_turn_tries_opus_first(self):
        # The window has passed. The next turn spends one call on Opus, which walls again, and moves on.
        agent = _public_home_agent()
        script = self._script(opus_rate_limit({"retry-after": "90"}))
        drive(agent, script)
        _expire_cooldown(agent)
        run = drive(agent, script)
        assert run.calls == [OPUS, KIMI]
        assert run.result["completed"] is True

    def test_a_wall_that_names_its_reset_is_not_tried_again_inside_the_window(self):
        agent = _public_home_agent()
        script = self._script(opus_rate_limit(unified_headers(time.time() + FOUR_HOURS)))
        drive(agent, script)
        assert FOUR_HOURS - 60 < _cooldown(agent) <= FOUR_HOURS
        assert drive(agent, script).calls == [KIMI]

    def test_a_wall_with_no_reset_signal_doubles_across_restores(self):
        # A restore zeroes the agent's count, and the route has served nothing since, so the shared
        # count keeps the doubling. A closed window with no reset signal costs fewer calls each time.
        agent = _public_home_agent()
        script = self._script(opus_rate_limit({}))
        drive(agent, script)
        assert 0 < _cooldown(agent) <= 60
        _expire_cooldown(agent)
        assert drive(agent, script).calls == [OPUS, KIMI]
        assert 60 < _cooldown(agent) <= 120


class _GatewaySessions:
    """The gateway's per-session agent cache, reduced to what decides reuse.

    ``finish_turn`` runs the real ``GatewayTurnMixin._run_agent_evict_on_fallback``, the call at the
    end of a gateway turn, and that method calls ``_evict_cached_agent`` here.
    """

    SESSION = "agent:main:synthetic:dm:1"

    def __init__(self, build):
        self._build = build
        self._cached = {}

    def agent(self, session=SESSION):
        if session not in self._cached:
            self._cached[session] = self._build()
        return self._cached[session]

    def finish_turn(self, agent, result, session=SESSION):
        from gateway.run_turn import GatewayTurnMixin

        turn = SimpleNamespace(session_key=session, agent_holder=[agent], result_holder=[result])
        GatewayTurnMixin._run_agent_evict_on_fallback(self, turn)

    def _evict_cached_agent(self, session_key):
        self._cached.pop(session_key, None)


class TestGatewayEviction:
    """The next turn on the gateway, Cade's live surface.

    ``_run_agent_evict_on_fallback`` drops the cached agent after a successful fallback turn so the
    next message retries the primary. The route's cooldown is in the shared store, so a fresh agent
    reads it and starts on the fallback. This is the F9 choice in step 6(d): the store, not a kept
    agent, because ``api_server`` and cron build an agent per run and a new session has no cached
    agent to keep.
    """

    CASES = [
        (_public_home_agent, lambda: opus_rate_limit({"retry-after": str(FOUR_HOURS)}), OPUS, KIMI),
        (_sol_home_agent, lambda: forwarded_sol_wall(FOUR_HOURS), SOL, OPUS),
        (_kimi_home_agent, lambda: kimi_rate_limit(retry_after=FOUR_HOURS), KIMI, OPUS),
    ]

    @pytest.mark.parametrize("build, wall, walled, served", CASES, ids=["public-home", "sol-forwarded-429", "ruled-kimi"])
    def test_an_eviction_keeps_the_cooldown_so_the_wall_is_not_paid_twice(self, build, wall, walled, served):
        sessions = _GatewaySessions(build)
        script = lambda route, attempt: wall() if route == walled else "answered on " + route[0]

        first = sessions.agent()
        notices = []
        first.status_callback = lambda kind, text: notices.append(text)
        run = drive(first, script)
        assert run.calls == [walled, served]
        assert run.result["completed"] is True
        assert FOUR_HOURS - 60 < _cooldown(first) <= FOUR_HOURS
        assert any("Primary retry eligible in ~14400 s" in text for text in notices), notices
        sessions.finish_turn(first, run.result)

        second = sessions.agent()
        assert second is not first
        assert _cooldown(second) == 0  # nothing yet: this agent has run no turn
        notices.clear()
        second.status_callback = first.status_callback
        run = drive(second, script)
        assert run.calls == [served]
        assert run.result["completed"] is True
        assert FOUR_HOURS - 60 < _cooldown(second) <= FOUR_HOURS
        assert any("Primary retry eligible in ~" in text for text in notices), notices

    def test_a_second_session_never_pays_the_first_sessions_wall(self):
        sessions = _GatewaySessions(_public_home_agent)
        wall = opus_rate_limit({"retry-after": str(FOUR_HOURS)})
        script = lambda route, attempt: wall if route == OPUS else "answered on kimi"
        assert drive(sessions.agent("a"), script).calls == [OPUS, KIMI]
        assert drive(sessions.agent("b"), script).calls == [KIMI]

    def test_a_per_run_agent_holds_the_cooldown(self):
        # ``api_server`` builds an agent per /v1/runs run, so each run here is a new agent.
        wall = opus_rate_limit({"retry-after": str(FOUR_HOURS)})
        script = lambda route, attempt: wall if route == OPUS else "answered on kimi"
        calls = [drive(_public_home_agent(), script).calls for _ in range(3)]
        assert calls == [[OPUS, KIMI], [KIMI], [KIMI]]

    def test_after_the_window_a_fresh_agent_tries_the_primary_and_it_clears(self):
        wall = opus_rate_limit({"retry-after": str(FOUR_HOURS)})
        drive(_public_home_agent(), lambda route, attempt: wall if route == OPUS else "answered on kimi")
        agent = _public_home_agent()
        _expire_cooldown(agent)
        run = drive(agent, lambda route, attempt: "answered on " + route[0])
        assert run.calls == [OPUS]
        assert _ledger(OPUS) is None

    def test_a_failed_turn_keeps_the_agent_and_its_cooldown(self):
        sessions = _GatewaySessions(_sol_home_agent)
        sol_wall = forwarded_sol_wall(FOUR_HOURS)
        opus_wall = opus_rate_limit({"retry-after": str(FOUR_HOURS)})
        kimi_wall = kimi_rate_limit(retry_after=60)
        walls = {SOL: sol_wall, OPUS: opus_wall, KIMI: kimi_wall}
        script = lambda route, attempt: walls[route]

        first = sessions.agent()
        run = drive(first, script)
        assert run.result["failed"] is True
        sessions.finish_turn(first, run.result)

        assert sessions.agent() is first
        assert FOUR_HOURS - 60 < _cooldown(first) <= FOUR_HOURS
        assert set(drive(first, script).calls) == {KIMI}


@pytest.mark.usefixtures("custody_gate")
class TestTurnStartHold:
    """``hold_walled_primary``: what a turn does when its primary starts in a shared cooldown."""

    @staticmethod
    def _arm(route, seconds=FOUR_HOURS):
        from agent.route_cooldown import record_route_cooldown

        record_route_cooldown(route[0], route[1], "", seconds=seconds, count=1, reason="rate_limit")

    def test_the_hold_walks_the_chain_in_order_past_a_second_closed_route(self):
        self._arm(KIMI)
        self._arm(OPUS)
        run = drive(_kimi_home_agent(), lambda route, attempt: "answered on " + route[0])
        assert run.calls == [SOL]
        assert _reply(run.result) == "answered on cade-siwc"

    def test_the_prompt_names_the_route_that_carries_the_turn(self):
        self._arm(KIMI)
        agent = _kimi_home_agent()
        agent._cached_system_prompt = "stored prompt\nModel: k3.1-pro\nProvider: kimi-coding"
        run = drive(agent, lambda route, attempt: "answered")
        assert agent._cached_system_prompt.endswith(f"Model: {OPUS[1]}\nProvider: {OPUS[0]}")
        # The request, not only the cache: the turn took its prompt before the hold moved the route.
        sent = str(run.requests[0].get("system"))
        assert f"Model: {OPUS[1]}" in sent
        assert "k3.1-pro" not in sent

    def test_with_no_other_route_open_the_primary_keeps_the_turn(self):
        # The deadlines can be wrong, so a closed chain does not refuse the turn.
        for route in (KIMI, OPUS, SOL):
            self._arm(route)
        run = drive(_kimi_home_agent(), lambda route, attempt: "answered on " + route[0])
        assert run.calls == [KIMI]
        assert run.result["completed"] is True

    def test_custody_refuses_a_candidate_at_the_hold(self, custody_gate):
        # The walk is the mid-turn walk, so the custody skip runs. Nothing is added to the session's
        # unavailable set: a refusal holds for this turn.
        self._arm(KIMI)
        agent = _kimi_home_agent()
        custody_gate.decide = lambda candidate: {"action": "block" if candidate["provider"] == OPUS[0] else "allow"}
        run = drive(agent, lambda route, attempt: "answered on " + route[0])
        assert run.calls == [SOL]
        assert not agent._unavailable_fallback_keys

    def test_a_real_gate_sees_this_turns_labels_and_the_required_flag_at_the_hold(self, custody_gate):
        self._arm(KIMI)
        self._arm(OPUS)
        run = drive(_kimi_home_agent(), lambda route, attempt: "answered on " + route[0])
        assert run.calls == [SOL]
        [sol] = custody_gate.judged(SOL)
        assert (sol["phase"], sol["required"], sol["labels"]) == ("failover", True, Gate.LABELS)

    def test_custody_refusing_every_candidate_leaves_the_primary_with_a_whole_chain(self, custody_gate):
        self._arm(KIMI)
        agent = _kimi_home_agent()
        custody_gate.decide = lambda candidate: {"action": "block"}
        run = drive(agent, lambda route, attempt: "answered on " + route[0])
        assert run.calls == [KIMI]
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False

    def test_a_raising_gate_refuses_every_candidate_and_the_primary_still_runs_unjudged(self, custody_gate):
        # F12 in cade#256: nothing judges the primary, so a client raw turn can reach Kimi against
        # SPR-102. This pins that gap as it stands. It is not the fix.
        def boom(candidate):
            raise RuntimeError("synthetic")

        self._arm(KIMI)
        custody_gate.decide = boom
        run = drive(_kimi_home_agent(), lambda route, attempt: "answered on " + route[0])
        assert run.calls == [KIMI]
        assert custody_gate.judged(KIMI) == []

    def test_labels_are_known_before_the_hold(self):
        # A hold ahead of ``turn_route`` would judge the turn without its labels.
        self._arm(KIMI)
        seen = []
        real = __import__("agent.route_hooks", fromlist=["begin_turn"]).begin_turn

        def begin(agent, **kwargs):
            seen.append(("begin", agent.provider))
            return real(agent, **kwargs)

        with patch("agent.turn_context.begin_route_turn", side_effect=begin):
            run = drive(_kimi_home_agent(), lambda route, attempt: "answered on " + route[0])
        assert seen == [("begin", KIMI[0])]
        assert run.calls == [OPUS]

    def test_a_fallback_kept_from_an_earlier_turn_is_left_alone(self):
        self._arm(KIMI)
        agent = _kimi_home_agent()
        script = lambda route, attempt: "answered on " + route[0]
        drive(agent, script)
        assert drive(agent, script).calls == [OPUS]
        assert agent._fallback_index == 1

    def test_a_route_with_no_chain_never_reads_the_store(self):
        self._arm(KIMI)
        agent = _kimi_home_agent(fallback=[])
        with patch("agent.fallback_cooldown.read_route_cooldown") as read:
            run = drive(agent, lambda route, attempt: "answered")
        assert run.calls == [KIMI]
        read.assert_not_called()


class TestSolWallThroughTheRelay:
    """A wall on the cade home's live route, in the shapes the relay sends today. SPR-301."""

    @staticmethod
    def _script(wall):
        return lambda route, attempt: wall if route == SOL else "answered on " + route[0]

    @pytest.mark.parametrize("wall", [relay_rejected(), relay_stream_broken()], ids=["502", "broken-stream"])
    def test_a_masked_wall_spends_the_retries_then_moves_to_opus(self, wall):
        # Hermes reads a transient failure. It retries Sol, moves at the end and arms no cooldown.
        agent = _sol_home_agent()
        run = drive(agent, self._script(wall))
        assert run.calls == [SOL] * agent._api_max_retries + [OPUS]
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on anthropic"
        assert agent.api_mode == "anthropic_messages"
        assert _cooldown(agent) == 0

    def test_a_masked_wall_costs_every_turn_the_same_retries(self):
        # With no cooldown the next turn restores Sol and pays the retries again.
        agent = _sol_home_agent()
        script = self._script(relay_rejected())
        drive(agent, script)
        run = drive(agent, script)
        assert run.calls == [SOL] * agent._api_max_retries + [OPUS]

    @pytest.mark.parametrize("code", ["subscription_sharing_usage_limit_exceeded", "usage_limit_reached"])
    def test_a_response_failed_usage_code_arms_the_cooldown(self, code):
        # The relay forwards this frame with its code. ``turn_response_check`` moves on the first
        # invalid response, so Sol gets one call. F7 reads the code as a rate limit. This frame
        # carries no reset, so the cooldown is the 60 s first step. The first code comes from a
        # fixture in cade ``relay.test.mjs``, not from a live wall.
        agent = _sol_home_agent()
        run = drive(agent, self._script(responses_failed(code)))
        assert run.calls == [SOL, OPUS]
        assert run.result["completed"] is True
        assert 0 < _cooldown(agent) <= 60
        assert drive(agent, self._script(responses_failed(code))).calls == [OPUS]

    def test_the_soft_failure_call_site_carries_the_frames_reset(self):
        agent = _sol_home_agent()
        failed = responses_failed("usage_limit_reached", resets_in_seconds=FOUR_HOURS)
        run = drive(agent, self._script(failed))
        assert run.calls == [SOL, OPUS]
        assert FOUR_HOURS - 60 < _cooldown(agent) <= FOUR_HOURS
        assert FOUR_HOURS - 60 < _shared_seconds(SOL) <= FOUR_HOURS

    def test_the_soft_failure_call_site_passes_a_classified_reset(self):
        # Step 6(b) alone, with the classifier stubbed. The mutation that drops the reason or the
        # reset at the first call site fails here.
        agent = _sol_home_agent()
        verdict = ClassifiedError(reason=FailoverReason.rate_limit)
        context = {"reset_at": time.time() + FOUR_HOURS}
        failed = responses_failed("anything")
        with patch("agent.turn_recovery.classify_codex_soft_failure", return_value=(verdict, context)):
            run = drive(agent, self._script(failed))
        assert run.calls == [SOL, OPUS]
        assert FOUR_HOURS - 60 < _cooldown(agent) <= FOUR_HOURS

    def test_a_soft_failure_that_is_not_a_quota_wall_arms_nothing(self):
        agent = _sol_home_agent()
        run = drive(agent, self._script(responses_failed("server_error")))
        assert run.calls == [SOL, OPUS]
        assert _cooldown(agent) == 0
        assert _ledger(SOL) is None

    def test_reauthentication_moves_at_once_with_no_cooldown(self):
        agent = _sol_home_agent()
        script = self._script(relay_reauthentication_required())
        run = drive(agent, script)
        assert run.calls == [SOL, OPUS]
        assert run.waits == []
        assert _cooldown(agent) == 0
        assert drive(agent, script).calls == [SOL, OPUS]

    def test_with_the_chain_rendered_today_a_masked_wall_parks_the_turn(self):
        # ``interim_model_blocks`` renders no fallback. A wall reads as an outage, so after the
        # retries the auto-recovery ladder repeats them, then the turn ends with Hermes' error text.
        agent = _sol_home_agent(fallback=[])
        cycles = agent._auto_recovery_cycles
        assert cycles == 5
        expected = agent._api_max_retries * (1 + cycles)
        run = drive(agent, lambda route, attempt: relay_rejected(), max_api_calls=expected + 1)
        assert run.calls == [SOL] * expected
        assert run.result["completed"] is False
        assert _reply(run.result).strip(), run.result


class TestRelayOutage:
    """SPR-301 covers an outage as well as a quota wall. Here the route is down, not walled."""

    def test_a_relay_that_refuses_connections_moves_to_opus(self):
        # A transport failure takes the eager fallback at the second retry count, one call sooner
        # than the masked wall. Nothing arms a cooldown, so each later turn starts on Sol again.
        agent = _sol_home_agent()
        run = drive(agent, lambda route, attempt: relay_down() if route == SOL else "answered on " + route[0])
        assert run.calls == [SOL, SOL, OPUS]
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on anthropic"
        assert _cooldown(agent) == 0

    def test_with_the_chain_rendered_today_an_outage_ends_the_turn_on_sol(self):
        # Three more calls than the 502's 18: ``try_recover_primary_transport`` rebuilds the client
        # once and restarts the retries before the auto-recovery ladder runs.
        agent = _sol_home_agent(fallback=[])
        cycles = agent._auto_recovery_cycles
        expected = agent._api_max_retries * (2 + cycles)
        run = drive(agent, lambda route, attempt: relay_down(), max_api_calls=expected + 1)
        assert run.calls == [SOL] * expected
        assert run.result["completed"] is False
        assert _reply(run.result).strip(), run.result

    def test_all_three_routes_transient_end_the_turn_with_a_short_cooldown(self):
        # A chain walked on a non-rate-limit failure arms 5 s. The auto-recovery ladder runs on the
        # last route, so Kimi takes the 18 calls that Sol takes with no chain.
        agent = _sol_home_agent()
        retries, cycles = agent._api_max_retries, agent._auto_recovery_cycles
        expected = [SOL] * retries + [OPUS] * retries + [KIMI] * (retries * (1 + cycles))
        run = drive(agent, lambda route, attempt: relay_rejected(), max_api_calls=len(expected) + 1)
        assert run.calls == expected
        assert run.result["completed"] is False
        assert _reply(run.result).strip(), run.result
        assert 0 < _cooldown(agent) <= 5


class TestForwardedSolWall:
    """The same wall once the relay forwards it as a 429 (step 5)."""

    def test_a_forwarded_wall_moves_at_once_and_arms_the_cooldown(self):
        agent = _sol_home_agent()
        wall = forwarded_sol_wall(FOUR_HOURS)
        script = lambda route, attempt: wall if route == SOL else "answered on " + route[0]
        run = drive(agent, script)
        assert run.calls == [SOL, OPUS]
        assert run.waits == []
        assert FOUR_HOURS - 60 < _cooldown(agent) <= FOUR_HOURS
        assert drive(agent, script).calls == [OPUS]


class TestThreeRoutes:
    """Sol, then Opus, then Kimi, with Sol's wall in the forwarded shape. Sol is live today, so
    these show the walls on the way to the ruled order. TestRuledOrder pins that order."""

    @staticmethod
    def _script(sol_wall, opus_wall, kimi=None):
        def script(route, attempt):
            if route == SOL:
                return sol_wall
            if route == OPUS:
                return opus_wall
            return kimi if kimi is not None else "answered on kimi"
        return script

    def test_two_walls_reach_kimi_with_no_wait(self):
        agent = _sol_home_agent()
        script = self._script(forwarded_sol_wall(FOUR_HOURS), opus_rate_limit({"retry-after": str(FOUR_HOURS)}))
        run = drive(agent, script)
        assert run.calls == [SOL, OPUS, KIMI]
        assert run.waits == []
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on kimi"

    def test_a_masked_sol_wall_still_reaches_kimi(self):
        agent = _sol_home_agent()
        run = drive(agent, self._script(relay_rejected(), opus_rate_limit({"retry-after": str(FOUR_HOURS)})))
        assert run.calls == [SOL] * agent._api_max_retries + [OPUS, KIMI]
        assert run.result["completed"] is True

    def test_every_hop_records_its_window(self):
        # The hop from Opus to Kimi records Opus' four hours in the shared store. Once Sol's short
        # window passes, the next turn pays Sol's wall and skips Opus. ``_rate_limited_until`` stays
        # the primary's: the Opus hop leaves it alone.
        agent = _sol_home_agent()
        script = self._script(forwarded_sol_wall(90), opus_rate_limit({"retry-after": str(FOUR_HOURS)}))
        drive(agent, script)
        assert 0 < _cooldown(agent) <= 90
        assert FOUR_HOURS - 60 < _shared_seconds(OPUS) <= FOUR_HOURS
        assert drive(agent, script).calls == [KIMI]
        _expire_cooldown(agent)
        assert drive(agent, script).calls == [SOL, KIMI]

    def test_after_sol_reopens_the_next_turn_returns_to_sol(self):
        agent = _sol_home_agent()
        drive(agent, self._script(forwarded_sol_wall(90), opus_rate_limit({"retry-after": str(FOUR_HOURS)})))
        _expire_cooldown(agent, SOL)
        run = drive(agent, self._script("answered on sol", "unused"))
        assert run.calls == [SOL]
        assert _reply(run.result) == "answered on sol"
        assert (agent.provider, agent.model, agent.api_mode) == (*SOL, "codex_responses")

    def test_a_kimi_wall_inside_sols_cooldown_does_not_return_to_opus(self):
        # The chain index stays past Opus while Sol's cooldown holds the restore. Kimi's wall ends
        # the turn on Kimi, even when Opus has reopened by then.
        agent = _sol_home_agent()
        drive(agent, self._script(forwarded_sol_wall(FOUR_HOURS), opus_rate_limit({"retry-after": "90"})))
        kimi_wall = kimi_rate_limit(retry_after=60)
        run = drive(agent, self._script("unused", "answered on opus", kimi_wall))
        assert set(run.calls) == {KIMI}
        assert run.result["completed"] is False

    def test_three_walls_end_the_turn_with_text(self):
        agent = _sol_home_agent()
        script = self._script(
            forwarded_sol_wall(FOUR_HOURS), opus_rate_limit({"retry-after": str(FOUR_HOURS)}),
            kimi_rate_limit(retry_after=FOUR_HOURS),
        )
        run = drive(agent, script)
        assert run.calls == [SOL, OPUS] + [KIMI] * agent._api_max_retries
        assert run.result["completed"] is False
        # As in TestBothWalls: Hermes' own text, after the capped waits on the last route.
        assert _reply(run.result).strip(), run.result
        assert _waits_on(run, SOL) == [] and _waits_on(run, OPUS) == []


def _ruled_script(*closed):
    walls = {
        KIMI: kimi_rate_limit(retry_after=FOUR_HOURS), OPUS: opus_rate_limit({"retry-after": str(FOUR_HOURS)}),
        SOL: forwarded_sol_wall(FOUR_HOURS), ONPREM: kimi_rate_limit(retry_after=FOUR_HOURS),
    }

    def script(route, attempt):
        return walls[route] if route in closed else "answered on " + route[0]
    return script


class TestRuledOrderWithNoGate:
    """The rendered Sol entry needs ``route_custody: required`` answered. With no plugin it carries
    nothing, which is what a home on #256 does until cade ships its gate plugin."""

    def test_a_required_sol_entry_is_never_called(self):
        agent = _kimi_home_agent()
        run = drive(agent, _ruled_script(KIMI, OPUS))
        assert run.calls == [KIMI] + [OPUS] * agent._api_max_retries
        assert run.result["completed"] is False
        assert _reply(run.result).strip(), run.result


@pytest.mark.usefixtures("custody_gate")
class TestRuledOrder:
    """J160(a): Kimi, then Opus, then Sol, with the on-prem tail J160(b) keeps for C1 and C2 turns."""

    _script = staticmethod(_ruled_script)

    def test_the_turn_starts_on_kimi(self):
        run = drive(_kimi_home_agent(), self._script())
        assert run.calls == [KIMI]

    def test_a_kimi_wall_moves_to_opus(self):
        agent = _kimi_home_agent()
        run = drive(agent, self._script(KIMI))
        assert run.calls == [KIMI, OPUS]
        assert run.waits == []
        assert _reply(run.result) == "answered on anthropic"
        assert agent.api_mode == "anthropic_messages"

    def test_two_walls_reach_sol_on_the_relays_api_mode(self, custody_gate):
        agent = _kimi_home_agent()
        run = drive(agent, self._script(KIMI, OPUS))
        assert run.calls == [KIMI, OPUS, SOL]
        assert run.waits == []
        assert _reply(run.result) == "answered on cade-siwc"
        # The entry names neither: both come from the rendered ``providers:`` block.
        assert (agent.provider, agent.base_url, agent.api_mode) == (SOL[0], SIWC_BASE_URL, "codex_responses")
        assert [kw["required"] for kw in custody_gate.judged(SOL)] == [True]
        assert [kw["required"] for kw in custody_gate.judged(OPUS)] == [False]

    def test_a_gate_refusal_keeps_sol_out_of_the_turn(self, custody_gate):
        custody_gate.decide = lambda candidate: {"action": "block" if candidate["provider"] == SOL[0] else "allow"}
        agent = _kimi_home_agent()
        run = drive(agent, self._script(KIMI, OPUS))
        assert run.calls == [KIMI] + [OPUS] * agent._api_max_retries
        assert run.result["completed"] is False

    def test_one_soft_wall_on_the_last_route_arms_one_window(self):
        # Both ``retry_invalid_response`` sites re-arm, so per arm this left count 4 and 480 seconds.
        agent = _kimi_home_agent()
        failed = responses_failed("usage_limit_reached")
        script = self._script(KIMI, OPUS)
        run = drive(agent, lambda route, attempt: failed if route == SOL else script(route, attempt))
        assert run.calls == [KIMI, OPUS] + [SOL] * agent._api_max_retries
        assert _ledger(SOL).count == 1
        assert 0 < _shared_seconds(SOL) <= 60

    def test_a_masked_sol_wall_as_the_last_route_ends_with_text(self):
        # Sol is the last route here, so the relay's 502 takes the auto-recovery ladder.
        agent = _kimi_home_agent()
        kimi_wall, opus_wall = kimi_rate_limit(retry_after=FOUR_HOURS), opus_rate_limit({"retry-after": str(FOUR_HOURS)})
        walls = {KIMI: kimi_wall, OPUS: opus_wall, SOL: relay_rejected()}
        retries, cycles = agent._api_max_retries, agent._auto_recovery_cycles
        run = drive(agent, lambda route, attempt: walls[route], max_api_calls=2 + retries * (1 + cycles) + 1)
        assert run.calls == [KIMI, OPUS] + [SOL] * (retries * (1 + cycles))
        assert run.result["completed"] is False
        assert _reply(run.result).strip(), run.result

    def test_three_walls_end_the_turn_with_text(self):
        agent = _kimi_home_agent()
        run = drive(agent, self._script(KIMI, OPUS, SOL))
        assert run.calls == [KIMI, OPUS] + [SOL] * agent._api_max_retries
        assert run.result["completed"] is False
        assert _reply(run.result).strip(), run.result

    def test_each_later_turn_goes_straight_to_the_open_route(self):
        script = self._script(KIMI, OPUS)
        assert drive(_kimi_home_agent(), script).calls == [KIMI, OPUS, SOL]
        assert drive(_kimi_home_agent(), script).calls == [SOL]
        assert drive(_kimi_home_agent(), script).calls == [SOL]

    def test_a_walled_tail_is_skipped_on_the_next_turn(self):
        # The agent records the tail with its resolved URL and the rendered entry has none, so both
        # must land on one file. A closed window then costs the tail no call on the next turn.
        script = self._script(KIMI, OPUS, SOL, ONPREM)
        first = _kimi_home_agent(fallback=RULED_CHAIN_WITH_TAIL)
        assert ONPREM in drive(first, script).calls
        assert FOUR_HOURS - 60 < _shared_seconds(ONPREM) <= FOUR_HOURS
        assert first.base_url == ONPREM_BASE_URL
        run = drive(_kimi_home_agent(fallback=RULED_CHAIN_WITH_TAIL), script)
        assert ONPREM not in run.calls

    def test_a_tail_that_is_not_rendered_ends_on_sol_and_the_status_reply(self, custody_gate):
        # #256 does not render a cold tail, so the chain ends on Sol and F6 answers.
        custody_gate.manager._hooks["on_routes_exhausted"] = [lambda **kw: {"reply": "Every model route is resting."}]
        agent = _kimi_home_agent()
        run = drive(agent, self._script(KIMI, OPUS, SOL))
        assert run.calls == [KIMI, OPUS] + [SOL] * agent._api_max_retries
        assert run.result["completed"] is False
        assert _reply(run.result) == "Every model route is resting."

    def test_a_rendered_tail_that_refuses_falls_through_to_the_status_reply(self, custody_gate):
        # J160(b): until a local model answers the turn reaches the SPR-151 reply. The reply text is
        # Cade's, through F6. A refused connection is a tail rendered warm that has gone cold.
        def cold(route, attempt):
            if route == ONPREM:
                return openai.APIConnectionError(request=httpx.Request("POST", f"{ONPREM_BASE_URL}/chat"))
            return script(route, attempt)

        script = self._script(KIMI, OPUS, SOL)
        exhausted = []
        custody_gate.manager._hooks["on_routes_exhausted"] = [
            lambda **kw: exhausted.append(kw) or {"reply": "Every model route is resting."}]
        agent = _kimi_home_agent(fallback=RULED_CHAIN_WITH_TAIL)
        retries, cycles = agent._api_max_retries, agent._auto_recovery_cycles
        run = drive(agent, cold, max_api_calls=3 + retries * (2 + cycles) + 1)
        assert run.calls[:4] == [KIMI, OPUS, SOL, ONPREM]
        assert set(run.calls[3:]) == {ONPREM}
        assert run.result["completed"] is False
        assert _reply(run.result) == "Every model route is resting."
        assert len(exhausted) == 1


class TestRouteReceipts:
    """F8 through the real loop: what a quota watch hears about each attempt."""

    @staticmethod
    def _receipts():
        from hermes_cli.plugins import PluginManager

        manager = PluginManager()
        manager._discovered = True
        seen = []
        manager._hooks["post_api_response"] = [lambda receipt, **kw: seen.append(receipt)]
        return patch("hermes_cli.plugins.get_plugin_manager", return_value=manager), seen

    def test_a_wall_then_a_served_fallback_gives_two_receipts(self):
        loaded, seen = self._receipts()
        wall = opus_rate_limit(unified_headers(time.time() + FOUR_HOURS))
        with loaded:
            drive(_public_home_agent(), lambda route, attempt: wall if route == OPUS else "answered on kimi")
        assert [(r["provider"], r["status"]) for r in seen] == [(OPUS[0], 429), (KIMI[0], 200)]
        assert seen[0]["headers"]["anthropic-ratelimit-unified-status"] == "rejected"
        assert seen[1]["headers"] == {}

    def test_a_soft_quota_wall_reports_429_and_not_200(self):
        # ingestReceipt treats a 2xx as a live success, which would close the breaker on a wall.
        loaded, seen = self._receipts()
        failed = responses_failed("usage_limit_reached")
        with loaded:
            drive(_sol_home_agent(), lambda route, attempt: failed if route == SOL else "answered on " + route[0])
        assert [(r["provider"], r["status"]) for r in seen] == [(SOL[0], 429), (OPUS[0], 200)]

    def test_a_refused_connection_has_no_receipt(self):
        loaded, seen = self._receipts()
        with loaded:
            drive(_sol_home_agent(), lambda route, attempt: relay_down() if route == SOL else "answered on " + route[0])
        assert [(r["provider"], r["status"]) for r in seen] == [(OPUS[0], 200)]

    def test_a_raising_observer_leaves_the_turn_alone(self):
        from hermes_cli.plugins import PluginManager

        def boom(**kw):
            raise RuntimeError("synthetic")

        manager = PluginManager()
        manager._discovered = True
        manager._hooks["post_api_response"] = [boom]
        with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
            run = drive(_public_home_agent(), lambda route, attempt: "answered")
        assert run.result["completed"] is True
