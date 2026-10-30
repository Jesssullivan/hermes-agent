"""Route hooks for an operator's route gate: turn labels, a custody skip at failover, and the
routes-exhausted reply.

Three plugin hooks let a deployment decide which model route may carry a turn. None of them
changes anything until a plugin registers for it, or a fallback entry sets ``route_custody:
required``.

- ``turn_route`` (F2) runs once per turn before turn-start compaction and the first API call
  (``phase="turn"``), and when a resolution-time walker is about to pick a fallback for a
  primary whose credentials failed (``phase="resolve"``). A callback may return
  ``{"labels": [...]}``. Labels are plugin-defined strings, sticky for the session id within
  this process, and handed to ``route_custody``. Hermes never interprets them.
- ``route_custody`` (F3) runs for every fallback candidate that passed the existing skips in
  ``_should_skip_fallback_candidate`` (``phase="failover"``), for every resolution-time
  candidate (``phase="resolve"``: gateway, one-shot, TUI, cron and the agent's init-time
  walker), for every main-chain or task-chain candidate of an auxiliary call
  (``phase="auxiliary"``), and at turn start (``phase="turn"``) when the agent sits on a
  fallback kept from an earlier turn or on a route that is an entry of its own fallback chain.
  ``{"action": "skip"}`` refuses the candidate for this turn only. The hook fails closed: a
  callback that raises or times out refuses the candidate. An entry with ``route_custody:
  required`` also needs an explicit ``{"action": "allow"}``; any other value that is not an
  off value counts as required. While no gate is loaded such an entry carries no request.
- ``on_routes_exhausted`` (F6) runs when a turn ends on a provider error with no fallback left,
  or when custody refuses the turn's only route. It is the event a gate pings from. A callback
  may return ``{"reply": "..."}`` (or a string) to replace the turn's user-facing copy for a
  capacity ending (rate limit, quota, overload, outage, timeout) or a custody refusal.

``HERMES_ROUTE_CAPABILITIES`` names what this build provides, keyed by the route-gate ids. F1 is
the pair of keys that keep auxiliary and compression traffic off the main model.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
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
# Values that leave an entry ungated. Any other value (``true``, ``yes``, a typo) counts as
# required, with a warning, so a misspelt gate fails closed instead of open.
_ROUTE_CUSTODY_OFF = frozenset({"", "optional", "off", "none", "no", "false"})
_REFUSE_ACTIONS = frozenset({"skip", "block", "deny"})
# Failure reasons whose copy an ``on_routes_exhausted`` reply may replace: capacity endings and a
# custody refusal. Auth, billing-wall, content-policy and request errors keep their own guidance.
_REPLY_REASONS = frozenset({"rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error",
                            "timeout", "route_custody"})
ROUTE_CUSTODY_REASON = "route_custody"
_MAX_LABELS = 64
_MAX_LABEL_CHARS = 128
_MAX_REPLY_CHARS = 2000
_MAX_SESSIONS = 1024
# Session labels by session id, so an agent rebuilt for the same session (gateway resolution-time
# fallback, cache eviction) and its auxiliary calls still see earlier turns' labels.
_SESSION_LABELS: OrderedDict[str, frozenset[str]] = OrderedDict()
_SESSION_LABELS_LOCK = threading.Lock()
_warned_values: set[str] = set()


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


def entry_requires_custody(entry: Mapping[str, Any]) -> bool:
    """Whether a fallback entry needs an explicit ``allow``. ``required`` does; an off value or a
    missing key does not; any other value does, with one warning per value (fail closed)."""
    raw = entry.get(ROUTE_CUSTODY_KEY)
    if raw is None or raw is False:
        return False
    value = str(raw).strip().lower()
    if value == ROUTE_CUSTODY_REQUIRED:
        return True
    if value in _ROUTE_CUSTODY_OFF:
        return False
    if value not in _warned_values:
        _warned_values.add(value)
        logger.warning("Route custody: unrecognised route_custody value %r on %s/%s; treating it as required",
                       raw, entry.get("provider"), entry.get("model"))
    return True


def _gated(entry: Mapping[str, Any]) -> bool:
    """Whether any custody decision applies to *entry*: a loaded gate or a custody key."""
    return _has_hook(ROUTE_CUSTODY_HOOK) or entry_requires_custody(entry)


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
                    session_id: str = "", turn_id: str = "", platform: str = "", task: str = "") -> bool:
    """Ask ``route_custody`` (F3) whether *entry* may carry this turn. Fails closed."""
    candidate = _route(entry.get("provider"), entry.get("model"))
    required = entry_requires_custody(entry)
    if not _has_hook(ROUTE_CUSTODY_HOOK):
        if required:
            logger.warning("Route custody: %s/%s needs route_custody and no plugin handles it; skipping",
                           candidate["provider"], candidate["model"])
        return required
    try:
        results = _invoke(
            ROUTE_CUSTODY_HOOK, phase=phase, candidate=candidate, current=dict(current),
            messages=list(messages or ()), data_use=sorted(data_use), labels=sorted(labels),
            required=required, session_id=session_id, turn_id=turn_id, platform=platform, task=task)
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


def session_labels(session_id: Any) -> frozenset[str]:
    """Labels ``turn_route`` gave *session_id* in this process (empty for an unknown or blank id)."""
    if not session_id:
        return frozenset()
    with _SESSION_LABELS_LOCK:
        return _SESSION_LABELS.get(str(session_id), frozenset())


def _remember_session_labels(session_id: Any, labels: frozenset[str]) -> None:
    if not session_id or not labels:
        return
    key = str(session_id)
    with _SESSION_LABELS_LOCK:
        _SESSION_LABELS[key] = _SESSION_LABELS.get(key, frozenset()) | labels
        _SESSION_LABELS.move_to_end(key)
        while len(_SESSION_LABELS) > _MAX_SESSIONS:
            _SESSION_LABELS.popitem(last=False)


def _session_labels(agent: Any) -> frozenset[str]:
    session_id = getattr(agent, "session_id", None)
    held = getattr(agent, "_route_labels", None)
    own = held[1] if isinstance(held, tuple) and len(held) == 2 and held[0] == session_id else frozenset()
    return own | session_labels(session_id)


def _add_session_labels(agent: Any, labels: frozenset[str]) -> None:
    if labels:
        session_id = getattr(agent, "session_id", None)
        agent._route_labels = (session_id, _session_labels(agent) | labels)
        _remember_session_labels(session_id, labels)


def _thread_session_id() -> str:
    """The session id ``set_session_context`` bound on this thread, for calls with no agent."""
    try:
        from hermes_logging import _session_context
        return str(getattr(_session_context, "session_id", "") or "")
    except ImportError:
        return ""


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


def fallback_custody_refuses(agent: Any, fb: Mapping[str, Any], fb_provider: str, fb_model: str, *,
                             phase: str = "failover") -> bool:
    """F3 at failover: ``True`` skips *fb* for this turn. Not added to the unavailable set: the
    next turn may carry different data."""
    if not _gated(fb):
        return False
    messages = _turn_messages(agent)
    return custody_refuses(
        {**fb, "provider": fb_provider, "model": fb_model}, phase=phase, current=_current_route(agent),
        messages=messages, data_use=_turn_data_use(agent, messages), labels=_session_labels(agent),
        **_agent_context(agent))


def begin_turn(agent: Any, *, messages: list, turn_id: str) -> None:
    """Turn start, before turn-start compaction and the first API call: record the turn, fire
    ``turn_route`` (F2), and re-check the active route when it is a fallback (F3). Raises
    :class:`RouteCustodyRefused` when custody refuses a route that cannot be left this turn."""
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
    if getattr(agent, "_provider_fallback_active", False):
        _recheck_kept_fallback(agent)
    else:
        _recheck_resolution_fallback(agent)


def _resolution_fallback_entry(agent: Any) -> Optional[Mapping[str, Any]]:
    """The fallback-chain entry the agent runs on when it was built on a fallback before it
    existed (gateway, one-shot, TUI, cron or init-time walker): its active route is an entry of
    its own chain. ``None`` when the active route is not in the chain."""
    current = _current_route(agent)
    for entry in getattr(agent, "_fallback_chain", None) or ():
        if isinstance(entry, Mapping) and _route(entry.get("provider"), entry.get("model")) == current:
            return entry
    return None


def _recheck_resolution_fallback(agent: Any) -> None:
    """F3 for a fallback picked before the agent existed: the resolve-phase gate saw no turn, so
    the turn's own messages and labels are checked here. The agent has no other primary to
    restore, so a refusal stops the turn before any request is sent."""
    entry = _resolution_fallback_entry(agent)
    if entry is None or not _gated(entry):
        return
    if fallback_custody_refuses(agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""), phase="turn"):
        raise RouteCustodyRefused(
            f"route custody refused {entry.get('provider')}/{entry.get('model')}, the route this agent "
            "was built on after its primary failed")


def _recheck_kept_fallback(agent: Any) -> None:
    """A fallback stays active across turns while the primary cools down. When custody refuses it
    for this turn, restore the primary now; its next failure walks the chain with the custody
    skip. If the primary cannot be restored, stop the turn rather than send on the refused route."""
    entry = getattr(agent, "_route_fallback_entry", None) or _resolution_fallback_entry(agent)
    if not isinstance(entry, Mapping):
        return
    if not fallback_custody_refuses(agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""), phase="turn"):
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
                       labels_cache: dict[str, frozenset[str]], session_id: str = "") -> bool:
    """F2 and F3 for a resolution-time walker, which runs before any agent or turn exists: the
    custody gate sees no messages and no data use, so it decides from labels: the resolve-phase
    labels plus, when the caller passes *session_id*, the labels earlier turns gave that session
    in this process. The thread's bound session is not used: a resolving thread may still carry
    another session's binding. The agent built on the chosen entry is checked again at each turn
    start with the turn's messages and the session's labels."""
    if not (_gated(entry) or _has_hook(TURN_ROUTE_HOOK)):
        return False
    session_id = str(session_id or "")
    if "labels" not in labels_cache:
        prior = session_labels(session_id)
        labels_cache["labels"] = prior | announce_route(
            phase="resolve", route=primary, chain=_chain_routes(chain), labels=prior, session_id=session_id,
            reason="auth")
    return custody_refuses(entry, phase="resolve", current=primary, labels=labels_cache["labels"],
                           session_id=session_id)


def auxiliary_refuses(entry: Mapping[str, Any], *, task: Optional[str], failed_provider: str,
                      failed_model: Optional[str] = None) -> bool:
    """F3 for an auxiliary call (compression, titles, vision, ...) walking the main or a task
    fallback chain. It has no agent or turn: the gate sees the task name and the labels of the
    session bound to this thread. A required entry is skipped while no gate is loaded."""
    if not _gated(entry):
        return False
    session_id = _thread_session_id()
    return custody_refuses(entry, phase="auxiliary", current=_route(failed_provider, failed_model),
                           labels=session_labels(session_id), session_id=session_id, task=str(task or ""))


def routes_exhausted(agent: Any, result: dict[str, Any], *, classified: Any, provider: Any,
                     model: Any, delivered: str = "") -> dict[str, Any]:
    """F6: fire ``on_routes_exhausted`` for a turn that ended on a provider error with no
    fallback left, and let a callback replace the user-facing copy of a capacity ending."""
    if not isinstance(result, dict) or not _has_hook(ROUTES_EXHAUSTED_HOOK):
        return result
    if getattr(agent, "_fallback_index", 0) < len(getattr(agent, "_fallback_chain", None) or []):
        return result
    context = getattr(classified, "error_context", None) or {}
    reset_at = context.get("reset_at") if isinstance(context, Mapping) else None
    reset_at = float(reset_at) if isinstance(reset_at, (int, float)) and not isinstance(reset_at, bool) else None
    reason = getattr(getattr(classified, "reason", None), "value", "") or ""
    return _fire_routes_exhausted(agent, result, reason=str(reason), retryable=bool(getattr(classified, "retryable", False)),
                                  reset_at=reset_at, provider=provider, model=model, delivered=delivered)


def _fire_routes_exhausted(agent: Any, result: dict[str, Any], *, reason: str, retryable: bool,
                           reset_at: Optional[float], provider: Any, model: Any, delivered: str = "") -> dict[str, Any]:
    primary = getattr(agent, "_primary_runtime", None) or {}
    messages = _turn_messages(agent)
    try:
        results = _invoke(
            ROUTES_EXHAUSTED_HOOK, route=_route(provider, model), reason=reason, retryable=retryable,
            reset_at=reset_at, reset_in_seconds=max(0.0, reset_at - time.time()) if reset_at else None,
            primary=_route(primary.get("provider"), primary.get("model")) if primary else _route(provider, model),
            chain=_chain_routes(getattr(agent, "_fallback_chain", None)),
            data_use=sorted(_turn_data_use(agent, messages)), labels=sorted(_session_labels(agent)),
            **_agent_context(agent))
    except Exception:  # health: allow BLE001 -- the default exhausted copy stands
        logger.warning("on_routes_exhausted hook dispatch failed", exc_info=True)
        return result
    result["routes_exhausted"] = True
    reply = next((r for r in (_reply_from(x) for x in results) if r), "")
    if not reply:
        return result
    if reason not in _REPLY_REASONS or result.get("free_tier"):
        # Auth, re-login, content-policy and free-tier copy carry the way forward; keep it.
        logger.info("on_routes_exhausted reply not applied: the %s copy keeps its own guidance", reason or "unknown")
        return result
    kept = (delivered or "").strip()
    result["final_response"] = f"{kept}\n\n{reply}" if kept and result.get("partial") else reply
    result["routes_exhausted_reply"] = True
    return result


def route_custody_refused_result(agent: Any, exc: RouteCustodyRefused, conversation_history: Any) -> dict[str, Any]:
    """Turn result when custody refused the only route at turn start: no request was sent and
    the user row is not persisted (as for a preflight timeout). Fires ``on_routes_exhausted``
    with ``reason="route_custody"`` so a gate can send its status reply."""
    logger.warning("Turn stopped before any request: %s", exc)
    from agent.agent_runtime_helpers import note_turn_persisted
    note_turn_persisted(agent)
    copy = "This turn was not sent: no model route is cleared to carry it right now. Try again later."
    result = {"final_response": copy, "messages": list(conversation_history or []), "completed": False,
              "api_calls": 0, "error": copy, "partial": True, "failed": True,
              "turn_exit_reason": ROUTE_CUSTODY_REASON, "failure_reason": ROUTE_CUSTODY_REASON,
              "failure_retryable": False}
    if not _has_hook(ROUTES_EXHAUSTED_HOOK):
        return result
    return _fire_routes_exhausted(agent, result, reason=ROUTE_CUSTODY_REASON, retryable=False, reset_at=None,
                                  provider=getattr(agent, "provider", ""), model=getattr(agent, "model", ""))


def _reply_from(result: Any) -> str:
    if isinstance(result, Mapping):
        result = result.get("reply")
    return result.strip()[:_MAX_REPLY_CHARS] if isinstance(result, str) else ""
