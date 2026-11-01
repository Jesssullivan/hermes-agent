"""``post_api_response`` (route-gate id F8, TIN-7765): one receipt per provider response.

A quota watch needs what the provider said about its windows, including on the success path, and
``post_api_request`` carries no headers. The receipt has the ``cade.route-receipt/v0`` shape that
xoxd-ai/cade ``src/quota-watch.ts`` ``parseRouteReceipt`` accepts: seven keys, rate headers from an
allowlist and token counts. It never holds a message, an error body or any other header, so an
``Authorization`` value has nowhere to arrive. ``provider`` is Hermes' own id. A plugin maps it onto
the names its consumer knows, and drops a route its consumer does not.

The observer does not decide anything. A hook failure never reaches the turn. A failed attempt with
no HTTP status (connection refused) has no receipt, because the schema requires a status.
"""

from __future__ import annotations

import contextlib
import logging
import re
from datetime import UTC, datetime
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

POST_API_RESPONSE_HOOK = "post_api_response"
ROUTE_RECEIPT_SCHEMA = "cade.route-receipt/v0"

_UNIFIED = "anthropic-ratelimit-unified-"
# quota-watch.ts RATE_HEADER_ALLOWLIST. A receipt with any other header is refused there.
ROUTE_RECEIPT_HEADERS = frozenset({
    *(_UNIFIED + name for name in (
        "status", "reset", "representative-claim", "5h-status", "5h-utilization", "5h-reset", "7d-status",
        "7d-utilization", "7d-reset", "fallback-percentage", "overage-status",
    )),
    "retry-after", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
})
# parseRouteReceipt's bound on a header value. One value it would refuse would lose the whole receipt.
_HEADER_VALUE = re.compile(r"[A-Za-z0-9_.:+-]{0,64}")


def receipt_headers(headers: Any) -> dict[str, str]:
    """The allowlisted rate headers of a response, lower-cased; ``{}`` when there are none."""
    kept: dict[str, str] = {}
    if not headers or not hasattr(headers, "items"):
        return kept
    with contextlib.suppress(Exception):
        for name, value in headers.items():
            lowered = str(name).lower()
            if lowered in ROUTE_RECEIPT_HEADERS and isinstance(value, str) and _HEADER_VALUE.fullmatch(value):
                kept[lowered] = value
    return kept


def _usage(summary: Optional[Mapping[str, Any]]) -> dict[str, int]:
    summary = summary or {}

    def _count(name: str) -> int:
        value = summary.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    return {
        "input": _count("input_tokens"), "output": _count("output_tokens"),
        "cacheRead": _count("cache_read_tokens"), "cacheWrite": _count("cache_write_tokens"),
    }


def _home() -> str:
    from hermes_constants import get_hermes_home, profile_name_for_home
    return profile_name_for_home(get_hermes_home()) or "default"


def build_route_receipt(*, provider: Any, status: int, headers: Any, usage: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": ROUTE_RECEIPT_SCHEMA, "home": _home(), "provider": str(provider or ""), "status": int(status),
        "headers": receipt_headers(headers), "usage": _usage(usage),
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }


def fire_route_receipt(agent: Any, *, status: Optional[int], task_id: str, turn_id: str, headers: Any = None,
                       response: Any = None) -> None:
    """Fire the observer. ``headers`` default to the rate headers the streaming clients captured
    for this attempt (``_last_route_headers``). Each receipt takes them, so none is read twice."""
    captured = getattr(agent, "_last_route_headers", None)
    agent._last_route_headers = None
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        return
    try:
        from hermes_cli import lifecycle
        if not lifecycle.has_hook(POST_API_RESPONSE_HOOK):
            return
        receipt = build_route_receipt(
            provider=getattr(agent, "provider", ""), status=status,
            headers=headers if headers is not None else captured,
            usage=agent._usage_summary_for_api_request_hook(response) if response is not None else None,
        )
        lifecycle.invoke_hook(
            POST_API_RESPONSE_HOOK, receipt=receipt, task_id=task_id, turn_id=turn_id,
            session_id=getattr(agent, "session_id", "") or "", platform=getattr(agent, "platform", "") or "",
            model=getattr(agent, "model", ""),
        )
    except Exception:
        logger.debug("post_api_response dispatch failed", exc_info=True)
