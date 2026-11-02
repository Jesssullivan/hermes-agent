"""Route custody within a turn: the loop's phase before each request, and the same check after a
tool round, before a post-tool compaction can send the new rows (``agent/route_hooks.py``). Nothing
here imports ``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
from typing import Any

from agent.route_hooks import ROUTE_CUSTODY_REASON, RouteCustodyRefused, check_request_route

logger = logging.getLogger("agent.conversation_loop")

ROUTE_CUSTODY_STOP_COPY = ("This turn stopped before its next request: no model route is cleared to carry its "
                           "data right now. Try again later.")


@dataclass
class RouteCheckVerdict:
    """``action`` is ``"break"`` when custody stopped the turn, else ``"fallthrough"``. The other
    fields are the loop locals the phase rebinds."""

    action: str
    active_system_prompt: Any
    api_call_count: Any
    failed: Any
    final_response: Any
    _turn_exit_reason: Any


def route_stops_turn(agent: Any, messages: Any, active_system_prompt: Any) -> tuple[Any, bool]:
    """``(active_system_prompt, stopped)`` after :func:`check_request_route`. A restore moved the
    agent to the first admitted route, so the main route is published again and the prompt synced.
    A check that cannot finish stops the turn (fail closed)."""
    try:
        if not check_request_route(agent, messages):
            return active_system_prompt, False
    except Exception as exc:  # health: allow BLE001 -- custody that cannot answer admits nothing
        logger.warning("Turn stopped before its next request: %s", exc)
        with suppress(Exception):
            agent._emit_diagnostic_status("❌ No model route is cleared to carry this turn's data")
        return active_system_prompt, True
    from agent.conversation_loop import _sync_failover_system_message
    from agent.turn_context import _publish_runtime_main
    _publish_runtime_main(agent)
    return _sync_failover_system_message(agent, None, active_system_prompt), False


def check_summary_route(agent: Any, messages: Any) -> None:
    """Before an attempt of the iteration-budget summary, a request like any other. Raises
    :class:`RouteCustodyRefused` when no route admits it."""
    if route_stops_turn(agent, messages, None)[1]:
        raise RouteCustodyRefused("no route admits the budget summary")


def enforce_request_route(
    agent: Any, *, messages: Any, active_system_prompt: Any, api_call_count: Any, failed: Any,
    final_response: Any, _turn_exit_reason: Any, refund: bool = True,
) -> RouteCheckVerdict:
    """Before each request of a turn. A stop on the first attempt refunds the iteration, whose
    request never left. A retry passes *refund* false: its first attempt did leave."""
    prompt, stopped = route_stops_turn(agent, messages, active_system_prompt)
    verdict = RouteCheckVerdict(action="fallthrough", active_system_prompt=prompt, api_call_count=api_call_count,
                                failed=failed, final_response=final_response, _turn_exit_reason=_turn_exit_reason)
    if stopped:
        verdict.action, verdict.failed = "break", True
        verdict.final_response, verdict._turn_exit_reason = ROUTE_CUSTODY_STOP_COPY, ROUTE_CUSTODY_REASON
        if refund:
            verdict.api_call_count -= 1
            agent._api_call_count = verdict.api_call_count
            with suppress(Exception):
                agent.iteration_budget.refund()
    return verdict
