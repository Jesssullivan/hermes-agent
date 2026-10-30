"""Route hooks for an operator's route gate: turn labels, a custody skip at failover, and the
routes-exhausted reply.

Three plugin hooks let a deployment decide which model route may carry a turn. None of them
changes anything until a plugin registers for it, or a fallback entry sets ``route_custody:
required``.

- ``turn_route`` (F2) runs once per turn before the first API call (``phase="turn"``) and when
  the resolution-time walker (``resolve_runtime_with_fallback``) is about to pick a fallback for
  a primary whose credentials failed (``phase="resolve"``). A callback may return
  ``{"labels": [...]}``. Labels are plugin-defined strings, sticky for the session, and handed
  to ``route_custody``. Hermes never interprets them.
- ``route_custody`` (F3) runs for every fallback candidate that passed the existing skips in
  ``_should_skip_fallback_candidate``, for every resolution-time candidate, and at turn start
  when the agent still sits on a fallback route. ``{"action": "skip"}`` refuses the candidate
  for this turn only. The hook fails closed: a callback that raises or times out refuses the
  candidate. An entry with ``route_custody: required`` also needs an explicit
  ``{"action": "allow"}``, so the entry is never used while no gate is loaded.
- ``on_routes_exhausted`` (F6) runs when a turn ends on a provider error with no fallback left.
  It is the event a gate pings from. A callback may return ``{"reply": "..."}`` (or a string)
  to replace the turn's user-facing copy.

``HERMES_ROUTE_CAPABILITIES`` names what this build provides, keyed by the route-gate ids. F1 is
the pair of keys that keep auxiliary and compression traffic off the main model.
"""

from __future__ import annotations

import logging
import time
import uuid
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

TURN_ROUTE_HOOK = "turn_route"
ROUTE_CUSTODY_HOOK = "route_custody"
ROUTES_EXHAUSTED_HOOK = "on_routes_exhausted"
ROUTE_HOOKS = (TURN_ROUTE_HOOK, ROUTE_CUSTODY_HOOK, ROUTES_EXHAUSTED_HOOK)

HERMES_ROUTE_CAPABILITIES: Mapping[str, str] = MappingProxyType({
    "F1": "auxiliary.main_model_fallback and compression.main_model_fallback",
    "F2": TURN_ROUTE_HOOK,
    "F3": ROUTE_CUSTODY_HOOK,
    "F6": ROUTES_EXHAUSTED_HOOK,
})

# Fallback-entry key: ``route_custody: required`` admits the entry only on an explicit allow.
ROUTE_CUSTODY_KEY = "route_custody"
ROUTE_CUSTODY_REQUIRED = "required"
_REFUSE_ACTIONS = frozenset({"skip", "block", "deny"})
_MAX_LABELS = 64
_MAX_LABEL_CHARS = 128
_MAX_REPLY_CHARS = 2000


class RouteCustodyRefused(RuntimeError):
    """The agent sits on a fallback route the custody gate refused, and the primary could not be
    restored. The turn stops before any request reaches the refused route."""


def _has_hook(name: str) -> bool:
    try:
        from hermes_cli.plugins import has_hook
        return bool(has_hook(name))
    except Exception:  # health: allow BLE001 -- plugin discovery failure means no subscriber
        logger.debug("Route hook %s: plugin lookup failed", name, exc_info=True)
        return False


def _invoke(name: str, **kwargs: Any) -> list[Any]:
    from hermes_cli.plugins import invoke_hook
    # ``call_id`` keeps concurrent sessions apart in the bounded-hook gate (plugins_dispatch).
    return invoke_hook(name, call_id=uuid.uuid4().hex, **kwargs)


def _route(provider: Any, model: Any) -> dict[str, str]:
    return {"provider": str(provider or "").strip().lower(), "model": str(model or "").strip()}


def _chain_routes(chain: Any) -> list[dict[str, str]]:
    return [_route(e.get("provider"), e.get("model")) for e in chain or () if isinstance(e, Mapping)]


def _clean_labels(raw: Any) -> frozenset[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    cleaned = sorted({s.strip()[:_MAX_LABEL_CHARS] for s in raw if isinstance(s, str) and s.strip()})
    return frozenset(cleaned[:_MAX_LABELS])


def _labels_from(results: Iterable[Any]) -> frozenset[str]:
    labels: set[str] = set()
    for result in results:
        if isinstance(result, Mapping):
            labels |= _clean_labels(result.get("labels"))
    return frozenset(labels)


def announce_route(*, phase: str, route: dict[str, str], chain: list[dict[str, str]],
                   messages: Optional[list] = None, data_use: Iterable[str] = (), labels: Iterable[str] = (),
                   session_id: str = "", turn_id: str = "", platform: str = "",
                   reason: str = "") -> frozenset[str]:
    """Fire ``turn_route`` (F2) and return the labels its callbacks added. Fails open: a callback
    that raises or times out adds no labels."""
    if not _has_hook(TURN_ROUTE_HOOK):
        return frozenset()
    try:
        results = _invoke(
            TURN_ROUTE_HOOK, phase=phase, route=dict(route), chain=list(chain), messages=list(messages or ()),
            data_use=sorted(data_use), labels=sorted(labels), session_id=session_id, turn_id=turn_id,
            platform=platform, reason=reason)
    except Exception:  # health: allow BLE001 -- an observer hook must never break the turn
        logger.warning("turn_route hook dispatch failed", exc_info=True)
        return frozenset()
    return _labels_from(results)


def custody_verdict(results: Iterable[Any], *, required: bool) -> tuple[bool, str]:
    """``(refused, reason)`` from ``route_custody`` results: any refusal wins; a required entry
    also needs an explicit allow."""
    allowed = False
    for result in results:
        action = result if isinstance(result, str) else result.get("action") if isinstance(result, Mapping) else None
        action = str(action or "").strip().lower()
        if action in _REFUSE_ACTIONS:
            message = (result.get("reason") or result.get("message")) if isinstance(result, Mapping) else ""
            return True, str(message or action)[:200]
        allowed = allowed or action == "allow"
    if required and not allowed:
        return True, "route_custody: required and no callback allowed the route"
    return False, ""


def custody_refuses(entry: Mapping[str, Any], *, phase: str, current: dict[str, str],
                    messages: Optional[list] = None, data_use: Iterable[str] = (), labels: Iterable[str] = (),
                    session_id: str = "", turn_id: str = "", platform: str = "") -> bool:
    """Ask ``route_custody`` (F3) whether *entry* may carry this turn. Fails closed."""
    candidate = _route(entry.get("provider"), entry.get("model"))
    required = str(entry.get(ROUTE_CUSTODY_KEY) or "").strip().lower() == ROUTE_CUSTODY_REQUIRED
    if not _has_hook(ROUTE_CUSTODY_HOOK):
        if required:
            logger.warning("Route custody: %s/%s needs route_custody and no plugin handles it; skipping",
                           candidate["provider"], candidate["model"])
        return required
    try:
        results = _invoke(
            ROUTE_CUSTODY_HOOK, phase=phase, candidate=candidate, current=dict(current),
            messages=list(messages or ()), data_use=sorted(data_use), labels=sorted(labels),
            required=required, session_id=session_id, turn_id=turn_id, platform=platform)
    except Exception:  # health: allow BLE001 -- a custody gate that cannot run refuses the route
        logger.warning("route_custody hook dispatch failed; refusing %s/%s", candidate["provider"],
                       candidate["model"], exc_info=True)
        return True
    refused, reason = custody_verdict(results, required=required)
    if refused:
        logger.warning("Route custody: %s/%s refused for this turn (%s, phase %s)",
                       candidate["provider"], candidate["model"], reason, phase)
    return refused


# -- agent-side wiring ---------------------------------------------------------------------------


def _session_labels(agent: Any) -> frozenset[str]:
    held = getattr(agent, "_route_labels", None)
    if isinstance(held, tuple) and len(held) == 2 and held[0] == getattr(agent, "session_id", None):
        return held[1]
    return frozenset()


def _add_session_labels(agent: Any, labels: frozenset[str]) -> None:
    if labels:
        agent._route_labels = (getattr(agent, "session_id", None), _session_labels(agent) | labels)


def _turn_messages(agent: Any) -> list:
    turn = getattr(agent, "_route_turn", None) or {}
    messages = turn.get("messages")
    live = getattr(agent, "_session_messages", None)
    out = list(messages) if isinstance(messages, list) else []
    if isinstance(live, list) and live is not messages:
        out.extend(live)
    return out


def _turn_data_use(agent: Any, messages: list) -> frozenset[str]:
    from agent.message_metadata import data_use_requirement
    turn = getattr(agent, "_route_turn", None) or {}
    return data_use_requirement(messages) | frozenset(turn.get("data_use") or ())


def _agent_context(agent: Any) -> dict[str, Any]:
    turn = getattr(agent, "_route_turn", None) or {}
    return {"session_id": str(getattr(agent, "session_id", "") or ""), "turn_id": str(turn.get("turn_id") or ""),
            "platform": str(getattr(agent, "platform", "") or "")}


def _current_route(agent: Any) -> dict[str, str]:
    return _route(getattr(agent, "provider", ""), getattr(agent, "model", ""))


def fallback_custody_refuses(agent: Any, fb: Mapping[str, Any], fb_provider: str, fb_model: str) -> bool:
    """F3 at failover: ``True`` skips *fb* for this turn. Not added to the unavailable set: the
    next turn may carry different data."""
    if not (_has_hook(ROUTE_CUSTODY_HOOK) or fb.get(ROUTE_CUSTODY_KEY)):
        return False
    messages = _turn_messages(agent)
    return custody_refuses(
        {**fb, "provider": fb_provider, "model": fb_model}, phase="failover", current=_current_route(agent),
        messages=messages, data_use=_turn_data_use(agent, messages), labels=_session_labels(agent),
        **_agent_context(agent))


def begin_turn(agent: Any, *, messages: list, turn_id: str) -> None:
    """Turn start, before the first API call: record the turn, fire ``turn_route`` (F2), and
    re-check a fallback route the agent kept from an earlier turn (F3)."""
    from agent.message_metadata import data_use_requirement
    gated = _has_hook(TURN_ROUTE_HOOK) or _has_hook(ROUTE_CUSTODY_HOOK)
    data_use = data_use_requirement(messages) if gated else frozenset()
    agent._route_turn = {"messages": messages, "turn_id": turn_id, "data_use": data_use}
    if gated and _has_hook(TURN_ROUTE_HOOK):
        labels = announce_route(
            phase="turn", route=_current_route(agent), chain=_chain_routes(getattr(agent, "_fallback_chain", None)),
            messages=messages, data_use=data_use, labels=_session_labels(agent),
            reason="fallback" if getattr(agent, "_provider_fallback_active", False) else "",
            **_agent_context(agent))
        _add_session_labels(agent, labels)
    _recheck_kept_fallback(agent)


def _recheck_kept_fallback(agent: Any) -> None:
    """A fallback stays active across turns while the primary cools down. When custody refuses it
    for this turn, restore the primary now; its next failure walks the chain with the custody
    skip. If the primary cannot be restored, stop the turn rather than send on the refused route."""
    entry = getattr(agent, "_route_fallback_entry", None)
    if not getattr(agent, "_provider_fallback_active", False) or not isinstance(entry, Mapping):
        return
    if not fallback_custody_refuses(agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or "")):
        return
    held_until = getattr(agent, "_rate_limited_until", 0) or 0
    agent._rate_limited_until = 0
    try:
        restored = agent._restore_primary_runtime()
    finally:
        agent._rate_limited_until = max(held_until, getattr(agent, "_rate_limited_until", 0) or 0)
    if not restored:
        raise RouteCustodyRefused(
            f"route custody refused the active fallback {entry.get('provider')}/{entry.get('model')} "
            "and the primary runtime could not be restored")
    agent._route_fallback_entry = None


def resolution_refuses(entry: Mapping[str, Any], *, primary: dict[str, str], chain: Iterable[Mapping[str, Any]],
                       labels_cache: dict[str, frozenset[str]]) -> bool:
    """F2 and F3 for the resolution-time walker, which runs before any agent or turn exists: the
    custody gate sees no messages and no data use, so it decides from labels alone."""
    if not (_has_hook(ROUTE_CUSTODY_HOOK) or _has_hook(TURN_ROUTE_HOOK) or entry.get(ROUTE_CUSTODY_KEY)):
        return False
    if "labels" not in labels_cache:
        labels_cache["labels"] = announce_route(phase="resolve", route=primary, chain=_chain_routes(chain), reason="auth")
    return custody_refuses(entry, phase="resolve", current=primary, labels=labels_cache["labels"])


def routes_exhausted(agent: Any, result: dict[str, Any], *, classified: Any, provider: Any,
                     model: Any) -> dict[str, Any]:
    """F6: fire ``on_routes_exhausted`` for a turn that ended on a provider error with no
    fallback left, and let a callback replace the user-facing copy."""
    if not isinstance(result, dict) or not _has_hook(ROUTES_EXHAUSTED_HOOK):
        return result
    if getattr(agent, "_fallback_index", 0) < len(getattr(agent, "_fallback_chain", None) or []):
        return result
    context = getattr(classified, "error_context", None) or {}
    reset_at = context.get("reset_at") if isinstance(context, Mapping) else None
    reset_at = float(reset_at) if isinstance(reset_at, (int, float)) and not isinstance(reset_at, bool) else None
    primary = getattr(agent, "_primary_runtime", None) or {}
    messages = _turn_messages(agent)
    reason = getattr(getattr(classified, "reason", None), "value", "") or ""
    try:
        results = _invoke(
            ROUTES_EXHAUSTED_HOOK, route=_route(provider, model), reason=str(reason),
            retryable=bool(getattr(classified, "retryable", False)), reset_at=reset_at,
            reset_in_seconds=max(0.0, reset_at - time.time()) if reset_at else None,
            primary=_route(primary.get("provider"), primary.get("model")) if primary else _route(provider, model),
            chain=_chain_routes(getattr(agent, "_fallback_chain", None)),
            data_use=sorted(_turn_data_use(agent, messages)), labels=sorted(_session_labels(agent)),
            **_agent_context(agent))
    except Exception:  # health: allow BLE001 -- the default exhausted copy stands
        logger.warning("on_routes_exhausted hook dispatch failed", exc_info=True)
        return result
    result["routes_exhausted"] = True
    reply = next((r for r in (_reply_from(x) for x in results) if r), "")
    if reply:
        result["final_response"] = reply
        result["routes_exhausted_reply"] = True
    return result


def _reply_from(result: Any) -> str:
    if isinstance(result, Mapping):
        result = result.get("reply")
    return result.strip()[:_MAX_REPLY_CHARS] if isinstance(result, str) else ""
