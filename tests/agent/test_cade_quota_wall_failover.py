"""Quota-wall failover harness for Cade's routes (TIN-7765, SPR-150 to SPR-153, SPR-301).

Two chains run through the real ``run_conversation`` loop with fake transports. These tests record
what this revision does at a quota wall.

The public-home chain is what the cade launcher renders at Anthropic stage ``active``: Opus 5.5 on
main turns with one Kimi Code fallback (xoxd-ai/cade ``scripts/hermes-profiles.py``
``public_model_blocks``).

The three-route chain is SPR-301: Sol through the SIWC relay, then Opus 5.5, then Kimi. Sol is the
cade home's live route. Cade does not render this chain: ``interim_model_blocks`` sets
``fallback_providers: []``. The chain here shows what Hermes would do with it. It says nothing about
custody. A cade-home Kimi entry also needs the F1, F2, F3 and F6 gate in
``docs/ROUTE_FAILOVER.md``, and these tests load no route hook.

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

import signal
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.codex_runtime import _raise_stream_error
from run_agent import AIAgent

SOL = ("cade-siwc", "gpt-6.1-sol")
OPUS = ("anthropic", "claude-opus-5-5")
KIMI = ("kimi-coding", "k3.1-pro")
# The public-home fallback block as cade renders it at stage active.
PUBLIC_FALLBACK = [{"provider": KIMI[0], "model": KIMI[1]}]
# SPR-301's three routes as one chain. Cade renders no such block today: the interim relay route has
# ``fallback_providers: []``. This is the chain under test, not a rendered config.
THREE_ROUTE_CHAIN = [{"provider": OPUS[0], "model": OPUS[1]}, {"provider": KIMI[0], "model": KIMI[1]}]
# The relay is a loopback listener. No test connects, so a reserved name stands in for its address.
SIWC_BASE_URL = "http://siwc-relay.invalid/v1"
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


def relay_stream_broken():
    """The frame the relay writes when a stream breaks after its headers went out."""
    try:
        _raise_stream_error({"type": "error", "code": "siwc_stream_not_completed"})
    except Exception as frame_error:
        return frame_error


def responses_failed(code):
    """A ``response.failed`` terminal. The relay forwards this frame, so its code reaches Hermes."""
    return SimpleNamespace(
        output=[], output_text="", status="failed", incomplete_details=None, error={"code": code},
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


def _route_client(provider, *args, **kwargs):
    """A fake of ``resolve_provider_client``: the configured fallback client, and no socket."""
    client = MagicMock()
    client.api_key = "synthetic"
    client.base_url = "https://api.anthropic.com" if provider == OPUS[0] else "https://api.kimi.com/coding/"
    client._custom_headers = None
    client.default_headers = None
    return client, kwargs.get("model")


class Run(SimpleNamespace):
    """One driven turn: result, calls as (provider, model) per API call, waits as (route, seconds)
    for each backoff Hermes asked for, and sleeps as any direct main-thread ``time.sleep`` argument."""


def drive(agent, script, max_api_calls=MAX_API_CALLS):
    """Run one turn. script(route, attempt) returns an exception to raise, the reply text for that
    call, or a whole response object."""
    calls, waits, sleeps = [], [], []

    def fake_api_call(api_kwargs, *args, **kwargs):
        if len(calls) >= max_api_calls:
            raise HarnessBound(f"more than {max_api_calls} API calls: {calls[-4:]}")
        route = (agent.provider, agent.model)
        calls.append(route)
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
    return Run(result=result, calls=calls, waits=waits, sleeps=sleeps)


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


def _expire_cooldown(agent):
    """Move the monotonic deadline into the past instead of waiting for it."""
    agent._rate_limited_until = time.monotonic() - 1


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
        # Only the anthropic-ratelimit-unified-* names, no Retry-After. This revision does not read
        # the unified reset (agent/retry_utils.py provider_retry_after_seconds reads Retry-After and
        # problem-detail fields), so the reset is unknown to Hermes. The turn still moves. Cade's quota
        # watch, not Hermes, has to carry the reset time into the status reply (F7 and F8).
        agent = _public_home_agent()
        wall = opus_rate_limit(unified_headers(time.time() + FOUR_HOURS))

        run = drive(agent, lambda route, attempt: wall if route == OPUS else "answered on kimi")
        assert run.calls == [OPUS, KIMI]
        assert run.result["completed"] is True
        assert _reply(run.result) == "answered on kimi"
        assert _opus_waits(run) == []

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
        assert _reply(run.result).strip() or run.result.get("error"), run.result
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
    """What the turn after a wall does.

    The cooldown is the rule. ``_arm_rate_limit_cooldown`` (agent/fallback_cooldown.py) runs when a
    rate-limit or billing failure leaves the primary. It takes the provider's reset, and with none it
    uses 60 s doubling to 4 h. ``restore_primary_runtime`` then keeps the agent on the fallback until
    that deadline passes, so a closed window costs no second 429 on the primary.
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

    def test_a_wall_with_no_reset_signal_stays_at_sixty_seconds(self):
        # Hermes does not read the unified reset, so a wall that names four hours arms 60 s. The
        # doubling never starts: the restore that lets Opus try again zeroes the backoff count. A
        # closed four-hour window therefore costs one Opus 429 per minute of turns.
        agent = _public_home_agent()
        script = self._script(opus_rate_limit(unified_headers(time.time() + FOUR_HOURS)))
        drive(agent, script)
        assert 0 < _cooldown(agent) <= 60
        _expire_cooldown(agent)
        assert drive(agent, script).calls == [OPUS, KIMI]
        assert 0 < _cooldown(agent) <= 60


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

    def test_a_response_failed_usage_code_arms_no_cooldown(self):
        # The relay forwards this frame with its code, and Hermes still treats it as transient.
        agent = _sol_home_agent()
        failed = responses_failed("subscription_sharing_usage_limit_exceeded")
        run = drive(agent, self._script(failed))
        assert set(run.calls[:-1]) == {SOL} and run.calls[-1] == OPUS
        assert run.result["completed"] is True
        assert _cooldown(agent) == 0

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
        assert _reply(run.result).strip() or run.result.get("error"), run.result


class TestForwardedSolWall:
    """The same wall once the relay forwards it as a 429. The relay does not do this today."""

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
    """Sol, then Opus, then Kimi, with Sol's wall in the forwarded shape."""

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

    def test_only_the_primary_gets_a_cooldown(self):
        # The hop from Opus to Kimi arms nothing, so Opus' four-hour window is not recorded. Once
        # Sol's short window passes, the next turn spends a call on each closed route again.
        agent = _sol_home_agent()
        script = self._script(forwarded_sol_wall(90), opus_rate_limit({"retry-after": str(FOUR_HOURS)}))
        drive(agent, script)
        assert 0 < _cooldown(agent) <= 90
        assert drive(agent, script).calls == [KIMI]
        _expire_cooldown(agent)
        assert drive(agent, script).calls == [SOL, OPUS, KIMI]

    def test_after_sol_reopens_the_next_turn_returns_to_sol(self):
        agent = _sol_home_agent()
        drive(agent, self._script(forwarded_sol_wall(90), opus_rate_limit({"retry-after": str(FOUR_HOURS)})))
        _expire_cooldown(agent)
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
        assert _reply(run.result).strip() or run.result.get("error"), run.result
        assert _waits_on(run, SOL) == [] and _waits_on(run, OPUS) == []
