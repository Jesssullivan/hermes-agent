"""Route hooks for an operator's route gate: turn labels, a custody skip at failover, and the
routes-exhausted reply.

Three plugin hooks let a deployment decide which model route may carry a turn. None of them
changes anything until a plugin registers for it, or a fallback entry sets ``route_custody:
required``.

- ``turn_route`` (F2) runs once per turn before turn-start compaction and the first API call
  (``phase="turn"``), and when a resolution-time walker is about to pick a fallback for a
  primary whose credentials failed (``phase="resolve"``). A callback may return
  ``{"labels": [...]}``. Labels are plugin-defined strings, sticky for the session within this
  process, and handed to ``route_custody``. Hermes never interprets them. A compaction that
  rotates the session id carries the labels and the data use to the child. The hook fails
  closed: when a callback raises or times out, the turn's labels are unknown, and every
  candidate a loaded ``route_custody`` gate would judge is refused for that turn.
- ``route_custody`` (F3) runs for every fallback candidate that passed the existing skips in
  ``_should_skip_fallback_candidate`` (``phase="failover"``), for every resolution-time
  candidate (``phase="resolve"``: gateway, one-shot, TUI, cron and the agent's init-time
  walker), for every main-chain or task-chain candidate of an auxiliary call
  (``phase="auxiliary"``), and at turn start (``phase="turn"``) when the agent sits on a
  fallback kept from an earlier turn or on the entry a resolution-time walker built it on.
  Each walker records that entry on the agent (:func:`bind_resolution_entry`); the turn-start
  check reads the record and never infers it from the agent's own chain. A compaction outside a
  turn on such a route is checked first (``phase="compaction"``) and binds its session for the
  auxiliary gate (:func:`enter_compaction_scope`, :func:`detached_compaction`).
  ``{"action": "skip"}`` refuses the candidate for this turn only. The hook fails closed: a
  callback that raises or times out refuses the candidate. An entry with ``route_custody:
  required`` also needs an explicit ``{"action": "allow"}``; any other value that is not an
  off value counts as required. While no gate is loaded such an entry carries no request.
- ``on_routes_exhausted`` (F6) runs when a turn ends on a provider error with no fallback left,
  or when custody refuses the turn's only route. It is the event a gate pings from. A callback
  may return ``{"reply": "..."}`` (or a string) to replace the turn's user-facing copy for a
  capacity ending (rate limit, overload, outage, timeout) or a custody refusal.

``HERMES_ROUTE_CAPABILITIES`` names what this build provides, keyed by the route-gate ids. F1 is
not listed. The ``auxiliary.main_model_fallback`` and ``compression.main_model_fallback`` keys
cover explicit auxiliary slots only: a slot left on ``provider: auto`` still walks the main
model and the main fallback chain.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

TURN_ROUTE_HOOK = "turn_route"
ROUTE_CUSTODY_HOOK = "route_custody"
ROUTES_EXHAUSTED_HOOK = "on_routes_exhausted"
ROUTE_HOOKS = (TURN_ROUTE_HOOK, ROUTE_CUSTODY_HOOK, ROUTES_EXHAUSTED_HOOK)

HERMES_ROUTE_CAPABILITIES: Mapping[str, str] = MappingProxyType({
    "F2": TURN_ROUTE_HOOK,
    "F3": ROUTE_CUSTODY_HOOK,
    "F6": ROUTES_EXHAUSTED_HOOK,
})

# Fallback-entry key: ``route_custody: required`` admits the entry only on an explicit allow.
ROUTE_CUSTODY_KEY = "route_custody"
ROUTE_CUSTODY_REQUIRED = "required"
# Runtime and agent-kwargs key a resolution-time walker stamps with the entry it chose. Callers pop
# it before an ``AIAgent(**kwargs)`` spread and hand it to :func:`bind_resolution_entry`.
ROUTE_ENTRY_KEY = "_route_fallback_entry"
# Values that leave an entry ungated. Any other value (``true``, ``yes``, a typo) counts as
# required, with a warning, so a misspelt gate fails closed instead of open.
_ROUTE_CUSTODY_OFF = frozenset({"", "optional", "off", "none", "no", "false"})
_REFUSE_ACTIONS = frozenset({"skip", "block", "deny"})
# Failure reasons whose copy an ``on_routes_exhausted`` reply may replace: capacity endings and a
# custody refusal. Auth, billing (402 credit exhaustion), content-policy and request errors keep
# their own guidance.
_REPLY_REASONS = frozenset({"rate_limit", "upstream_rate_limit", "overloaded", "server_error", "timeout",
                            "route_custody"})
ROUTE_CUSTODY_REASON = "route_custody"
_MAX_LABELS = 64
_MAX_LABEL_CHARS = 128
_MAX_REPLY_CHARS = 2000
_MAX_SESSIONS = 1024
# Labels and data use by session id, so an agent rebuilt for the same session (gateway
# resolution-time fallback, cache eviction) and its auxiliary calls still see earlier turns'.
_SESSIONS: OrderedDict[str, tuple[frozenset[str], frozenset[str]]] = OrderedDict()
_SESSIONS_LOCK = threading.Lock()
_warned_values: set[str] = set()
# The agent whose turn this context serves. ``begin_turn`` binds it; a worker started through
# ``propagate_context_to_thread`` (the compression pool) copies it, a thread-local would not.
_ROUTE_SCOPE: ContextVar[Optional[Callable[[], Any]]] = ContextVar("hermes_route_scope", default=None)
# The messages of the auxiliary request in flight, for the data use its custody check reports.
_AUX_REQUEST: ContextVar[Optional[list]] = ContextVar("hermes_route_aux_request", default=None)
# The messages a compaction summarizes. Out of a turn they are the only record of the session's data.
_COMPACTION_SOURCE: ContextVar[Optional[list]] = ContextVar("hermes_route_compaction_source", default=None)
# The entry record of the latest runtime resolution in this context. The gateway resolves in each
# turn's own copied context, so concurrent sessions never read each other's record.
_RESOLVED_ENTRY: ContextVar[Optional[Mapping[str, Any]]] = ContextVar("hermes_route_resolved_entry", default=None)
# The session a caller is resolving a runtime for, bound only for that resolution (:func:`resolving_session`).
_RESOLVING_SESSION: ContextVar[str] = ContextVar("hermes_route_resolving_session", default="")


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


def _labels_from(results: Iterable[Any]) -> tuple[frozenset[str], bool]:
    """``(labels, known)`` from ``turn_route`` results. ``turn_route`` is a fail-closed hook: a
    callback that raised or timed out yields a block directive, so the labels are unknown."""
    labels: set[str] = set()
    known = True
    for result in results:
        if isinstance(result, Mapping):
            if str(result.get("action") or "").strip().lower() in _REFUSE_ACTIONS:
                known = False
                logger.warning("turn_route callback failed (%s); labels unknown for this turn",
                               str(result.get("message") or "")[:200])
            labels |= _clean_labels(result.get("labels"))
    return frozenset(labels), known


def announce_route(*, phase: str, route: dict[str, str], chain: list[dict[str, str]],
                   messages: Optional[list] = None, data_use: Iterable[str] = (), labels: Iterable[str] = (),
                   session_id: str = "", turn_id: str = "", platform: str = "",
                   reason: str = "") -> tuple[frozenset[str], bool]:
    """Fire ``turn_route`` (F2): ``(labels its callbacks added, whether every callback answered)``.
    Fails closed: a callback that raises or times out, or a dispatch failure, makes the labels
    unknown, and custody then refuses every candidate it would judge."""
    if not _has_hook(TURN_ROUTE_HOOK):
        return frozenset(), True
    try:
        results = _invoke(
            TURN_ROUTE_HOOK, phase=phase, route=dict(route), chain=list(chain), messages=list(messages or ()),
            data_use=sorted(data_use), labels=sorted(labels), session_id=session_id, turn_id=turn_id,
            platform=platform, reason=reason)
    except Exception:  # health: allow BLE001 -- a labeller that cannot run leaves the labels unknown
        logger.warning("turn_route hook dispatch failed; labels unknown for this turn", exc_info=True)
        return frozenset(), False
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
                    session_id: str = "", turn_id: str = "", platform: str = "", task: str = "",
                    labels_known: bool = True) -> bool:
    """Ask ``route_custody`` (F3) whether *entry* may carry this turn. Fails closed, and refuses
    without asking when ``turn_route`` failed for this turn (*labels_known* false)."""
    candidate = _route(entry.get("provider"), entry.get("model"))
    required = entry_requires_custody(entry)
    if not _has_hook(ROUTE_CUSTODY_HOOK):
        if required:
            logger.warning("Route custody: %s/%s needs route_custody and no plugin handles it; skipping",
                           candidate["provider"], candidate["model"])
        return required
    if not labels_known:
        logger.warning("Route custody: %s/%s refused for this turn (turn_route failed, labels unknown, phase %s)",
                       candidate["provider"], candidate["model"], phase)
        return True
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


# -- session labels and data use -----------------------------------------------------------------


def _session_state(session_id: Any) -> tuple[frozenset[str], frozenset[str]]:
    if not session_id:
        return frozenset(), frozenset()
    with _SESSIONS_LOCK:
        return _SESSIONS.get(str(session_id), (frozenset(), frozenset()))


def session_labels(session_id: Any) -> frozenset[str]:
    """Labels ``turn_route`` gave *session_id* in this process (empty for an unknown or blank id)."""
    return _session_state(session_id)[0]


def session_data_use(session_id: Any) -> frozenset[str]:
    """Data use stamped on any turn of *session_id* in this process. It only grows: a compaction
    summary keeps the content its stamped rows held."""
    return _session_state(session_id)[1]


def _remember_session(session_id: Any, *, labels: frozenset[str] = frozenset(),
                      data_use: frozenset[str] = frozenset()) -> None:
    if not session_id or not (labels or data_use):
        return
    key = str(session_id)
    with _SESSIONS_LOCK:
        old_labels, old_data_use = _SESSIONS.get(key, (frozenset(), frozenset()))
        _SESSIONS[key] = (old_labels | labels, old_data_use | data_use)
        _SESSIONS.move_to_end(key)
        while len(_SESSIONS) > _MAX_SESSIONS:
            _SESSIONS.popitem(last=False)


def _remember_session_labels(session_id: Any, labels: frozenset[str]) -> None:
    _remember_session(session_id, labels=labels)


def _held(agent: Any, attr: str, session_id: Any) -> frozenset[str]:
    """The agent's own copy for *session_id* (kept past the registry's eviction)."""
    held = getattr(agent, attr, None)
    return held[1] if isinstance(held, tuple) and len(held) == 2 and held[0] == session_id else frozenset()


def _session_labels(agent: Any) -> frozenset[str]:
    session_id = getattr(agent, "session_id", None)
    return _held(agent, "_route_labels", session_id) | session_labels(session_id)


def _session_data_use(agent: Any) -> frozenset[str]:
    session_id = getattr(agent, "session_id", None)
    return _held(agent, "_route_data_use", session_id) | session_data_use(session_id)


def _grow_session(agent: Any, *, labels: frozenset[str] = frozenset(), data_use: frozenset[str] = frozenset()) -> None:
    session_id = getattr(agent, "session_id", None)
    if labels:
        agent._route_labels = (session_id, _session_labels(agent) | labels)
    if data_use:
        agent._route_data_use = (session_id, _session_data_use(agent) | data_use)
    _remember_session(session_id, labels=labels, data_use=data_use)


def _add_session_labels(agent: Any, labels: frozenset[str]) -> None:
    _grow_session(agent, labels=labels)


def carry_session_route_state(agent: Any, old_session_id: Any) -> None:
    """A compaction moved *agent* to a child session id: carry the parent's labels and data use to
    the child. Both only grow, and the summary rows carry no ``data_use`` stamp of their own."""
    labels = _held(agent, "_route_labels", old_session_id) | session_labels(old_session_id)
    data_use = _held(agent, "_route_data_use", old_session_id) | session_data_use(old_session_id)
    _grow_session(agent, labels=labels, data_use=data_use)


# -- agent-side wiring ---------------------------------------------------------------------------


def _scope_ref(agent: Any) -> Callable[[], Any]:
    try:
        return weakref.ref(agent)
    except TypeError:  # a stand-in without weakref support
        return lambda: agent


def _bind_scope(agent: Any) -> None:
    _ROUTE_SCOPE.set(_scope_ref(agent))


def _scope_agent() -> Any:
    ref = _ROUTE_SCOPE.get()
    return ref() if ref is not None else None


@contextmanager
def auxiliary_request(messages: Any) -> Iterator[None]:
    """Bind an auxiliary request's messages while its call runs, so a custody check in its
    fallback walk reports the data use stamped on them."""
    token = _AUX_REQUEST.set(messages if isinstance(messages, list) else None)
    try:
        yield
    finally:
        _AUX_REQUEST.reset(token)


def enter_compaction_scope(agent: Any, messages: Any) -> tuple[Any, Any]:
    """Bind *agent* as the route scope of a compaction and the *messages* it summarizes. Out of a
    turn no scope is bound (the gateway runs each compaction in a fresh copied context), so the
    auxiliary gate on any worker the compaction starts would see no session. Returns the tokens
    for :func:`exit_compaction_scope`; the first is ``None`` when *agent* already held the scope."""
    scope_token = None if _scope_agent() is agent else _ROUTE_SCOPE.set(_scope_ref(agent))
    return scope_token, _COMPACTION_SOURCE.set(messages if isinstance(messages, list) else None)


def exit_compaction_scope(tokens: tuple[Any, Any]) -> None:
    scope_token, source_token = tokens
    _COMPACTION_SOURCE.reset(source_token)
    if scope_token is not None:
        _ROUTE_SCOPE.reset(scope_token)


class _CompactionRoute:
    """The route and session of a compaction that runs with no agent (``session.mutate compress``)."""

    def __init__(self, *, session_id: str, provider: str, model: str, platform: str = "") -> None:
        self.session_id, self.provider, self.model, self.platform = session_id, provider, model, platform


@contextmanager
def detached_compaction(record: Any, *, session_id: str, provider: str, model: str, messages: list,
                        platform: str = "") -> Iterator[None]:
    """Route scope and custody for a compaction that summarizes *messages* on a resolved runtime
    with no agent: the auxiliary gate sees the session's labels and the messages' data use, and a
    gated resolution-time *record* is checked first. Raises :class:`RouteCustodyRefused`."""
    route = _CompactionRoute(session_id=str(session_id or ""), provider=str(provider or ""), model=str(model or ""),
                             platform=platform)
    bind_resolution_entry(route, record)
    tokens = enter_compaction_scope(route, messages)
    try:
        check_out_of_turn_route(route)
        yield
    finally:
        exit_compaction_scope(tokens)


def _turn_messages(agent: Any) -> list:
    turn = getattr(agent, "_route_turn", None) or {}
    messages = turn.get("messages")
    live = getattr(agent, "_session_messages", None)
    out = list(messages) if isinstance(messages, list) else []
    if isinstance(live, list) and live is not messages:
        out.extend(live)
    source = _COMPACTION_SOURCE.get()
    # Out of a turn the agent may hold no messages: the transcript being compacted stands in.
    return out or (list(source) if isinstance(source, list) else [])


def _compaction_data_use() -> frozenset[str]:
    from agent.message_metadata import data_use_requirement
    source = _COMPACTION_SOURCE.get()
    return data_use_requirement(source) if isinstance(source, list) else frozenset()


def _turn_data_use(agent: Any, messages: list) -> frozenset[str]:
    from agent.message_metadata import data_use_requirement
    turn = getattr(agent, "_route_turn", None) or {}
    return (data_use_requirement(messages) | frozenset(turn.get("data_use") or ()) | _session_data_use(agent)
            | _compaction_data_use())


def _labels_known(agent: Any) -> bool:
    turn = getattr(agent, "_route_turn", None) or {}
    return turn.get("labels_known", True) is not False


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
        labels_known=_labels_known(agent), **_agent_context(agent))


def begin_turn(agent: Any, *, messages: list, turn_id: str) -> bool:
    """Turn start, before turn-start compaction and the first API call: record the turn, fire
    ``turn_route`` (F2), and re-check the active route when it is a fallback (F3). Returns
    ``True`` when custody moved the agent back to its primary, so the caller publishes the new
    main route again. Raises :class:`RouteCustodyRefused` when custody refuses a route that
    cannot be left this turn."""
    from agent.message_metadata import data_use_requirement
    _bind_scope(agent)
    gated = _has_hook(TURN_ROUTE_HOOK) or _has_hook(ROUTE_CUSTODY_HOOK)
    data_use = data_use_requirement(messages) if gated else frozenset()
    agent._route_turn = {"messages": messages, "turn_id": turn_id, "data_use": data_use, "labels_known": True}
    if gated:
        _grow_session(agent, data_use=data_use)
    if gated and _has_hook(TURN_ROUTE_HOOK):
        labels, known = announce_route(
            phase="turn", route=_current_route(agent), chain=_chain_routes(getattr(agent, "_fallback_chain", None)),
            messages=messages, data_use=_turn_data_use(agent, messages), labels=_session_labels(agent),
            reason="fallback" if getattr(agent, "_provider_fallback_active", False) else "",
            **_agent_context(agent))
        agent._route_turn["labels_known"] = known
        _grow_session(agent, labels=labels)
    restored = False
    if getattr(agent, "_provider_fallback_active", False):
        restored = _recheck_kept_fallback(agent)
    if not getattr(agent, "_provider_fallback_active", False):
        # Also after a restore: the restored primary may itself be a resolution-time entry.
        _recheck_resolution_fallback(agent)
    return restored


# -- resolution-time entries ---------------------------------------------------------------------


def resolution_record(entry: Mapping[str, Any], runtime: Optional[Mapping[str, Any]] = None, *,
                      model: Any = None) -> dict[str, Any]:
    """The custody-relevant part of the fallback entry a resolution-time walker chose: its route
    as configured, the provider and model the walker resolved, and its custody key. No secret."""
    return {"provider": str(entry.get("provider") or "").strip(), "model": str(entry.get("model") or "").strip(),
            ROUTE_CUSTODY_KEY: entry.get(ROUTE_CUSTODY_KEY),
            "resolved_provider": str((runtime or {}).get("provider") or "").strip(),
            "resolved_model": str(model or "").strip()}


def mark_resolution_entry(runtime: dict[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """Stamp a resolved *runtime* with the fallback entry it came from, under :data:`ROUTE_ENTRY_KEY`."""
    runtime[ROUTE_ENTRY_KEY] = resolution_record(entry, runtime)
    return runtime


def stash_resolution_entry(record: Any) -> None:
    """Hold the entry record of the resolution just made (``None`` when the primary resolved) for
    the caller that builds on it, in this context only."""
    _RESOLVED_ENTRY.set(dict(record) if isinstance(record, Mapping) else None)


def take_resolution_entry() -> Optional[dict[str, Any]]:
    """The record :func:`stash_resolution_entry` held in this context, once."""
    record = _RESOLVED_ENTRY.get()
    _RESOLVED_ENTRY.set(None)
    return dict(record) if isinstance(record, Mapping) else None


def _bound_route(agent: Any) -> dict[str, str]:
    """The route the agent was built on: its primary while a kept fallback is active."""
    primary = getattr(agent, "_primary_runtime", None)
    if getattr(agent, "_provider_fallback_active", False) and isinstance(primary, Mapping):
        return _route(primary.get("provider"), primary.get("model"))
    return _current_route(agent)


def bind_resolution_entry(agent: Any, record: Any, *, source: str = "resolve") -> None:
    """Record on *agent* the fallback entry a resolution-time walker built it on, so each turn
    start re-checks custody for it. The record keeps the agent's own route as built
    (``bound_route``), so a model the agent rewrote after resolution still matches. A ``None``
    record clears an earlier record of the same *source* (the gateway resolves again every turn).
    The record does not depend on the agent's own chain, which a safe-mode or API-locked session
    leaves empty."""
    current = getattr(agent, "_route_resolution_entry", None)
    if isinstance(record, Mapping) and (record.get("provider") or record.get("resolved_provider")):
        bound = {**record, "source": source}
        _snapshot_bound_route(agent, bound)
        agent._route_resolution_entry = bound
    elif isinstance(current, Mapping) and current.get("source") == source:
        agent._route_resolution_entry = None


def _snapshot_bound_route(agent: Any, record: dict[str, Any]) -> None:
    """Keep the agent's route as ``bound_route`` when its provider is the entry's: a model the agent
    rewrote (``nous/welcome``) then still matches. A route on another provider is not the entry."""
    bound = _bound_route(agent)
    entry_forms = _provider_forms(record.get("provider")) | _provider_forms(record.get("resolved_provider"))
    if entry_forms & _provider_forms(bound["provider"]):
        record["bound_route"] = bound


def settle_init_resolution_route(agent: Any) -> None:
    """Agent construction finished: the init-time record takes the route the agent will send on,
    once. A later deliberate model switch leaves it alone and so reads as a move off the entry."""
    record = getattr(agent, "_route_resolution_entry", None)
    if isinstance(record, dict) and record.get("source") == "init" and not record.get("settled"):
        _snapshot_bound_route(agent, record)
        record["settled"] = True


def _provider_forms(name: Any) -> frozenset[str]:
    """*name* and its canonical ids, through both alias tables the stores use: the resolver's
    (``hermes_cli.auth``: ``claude`` to ``anthropic``, ``ollama`` to ``custom``) and the
    provider registry's (``hermes_cli.providers``: also ``openai`` to ``openrouter``)."""
    raw = str(name or "").strip().lower()
    if not raw:
        return frozenset()
    forms = {raw}
    with suppress(Exception):
        from hermes_cli.providers import normalize_provider
        forms.add(normalize_provider(raw))
    with suppress(Exception):
        from hermes_cli.auth import _plugin_aliases
        forms.add(_plugin_aliases().get(raw, raw))
    return frozenset(forms)


def _same_model(wanted: Iterable[str], model: str, provider: str) -> bool:
    wanted = {w for w in wanted if w}
    if model in wanted:
        return True
    with suppress(Exception):
        from hermes_cli.model_normalize import normalize_model_for_provider
        return any(normalize_model_for_provider(w, provider) == model for w in wanted)
    return False


def _resolution_fallback_entry(agent: Any) -> Optional[Mapping[str, Any]]:
    """The recorded resolution-time entry when a custody decision applies to it and the agent
    still runs on it. The agent's route matches the entry as configured, as resolved, or as the
    agent was built on it (``bound_route``), with provider names in canonical form on every side.
    ``None`` with no record, no gate, or after the agent moved off the entry (``/model``, a request
    route), which is logged once at warning."""
    record = getattr(agent, "_route_resolution_entry", None)
    if not isinstance(record, Mapping) or not _gated(record):
        return None
    current = _current_route(agent)
    bound = record.get("bound_route") if isinstance(record.get("bound_route"), Mapping) else {}
    wanted = (_provider_forms(record.get("provider")) | _provider_forms(record.get("resolved_provider"))
              | _provider_forms(bound.get("provider")))
    models = (str(record.get("model") or ""), str(record.get("resolved_model") or ""), str(bound.get("model") or ""))
    if wanted & _provider_forms(current["provider"]) and _same_model(models, current["model"], current["provider"]):
        return record
    if isinstance(record, dict) and not record.get("left_warned"):
        record["left_warned"] = True
        logger.warning("Route custody: the agent left the resolution-time entry %s/%s (now %s/%s); "
                       "turn starts no longer check it", record.get("provider"), record.get("model"),
                       current["provider"], current["model"])
    return None


def _recheck_resolution_fallback(agent: Any) -> None:
    """F3 for a fallback picked before the agent existed: the resolve-phase gate saw no turn, so
    the turn's own messages and labels are checked here. The agent has no other primary to
    restore, so a refusal stops the turn before any request is sent."""
    entry = _resolution_fallback_entry(agent)
    if entry is None:
        return
    if fallback_custody_refuses(agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""), phase="turn"):
        raise RouteCustodyRefused(
            f"route custody refused {entry.get('provider')}/{entry.get('model')}, the route this agent "
            "was built on after its primary failed")


def _recheck_kept_fallback(agent: Any) -> bool:
    """A fallback stays active across turns while the primary cools down. When custody refuses it
    for this turn, restore the primary now (``True``); its next failure walks the chain with the
    custody skip. If the primary cannot be restored, stop the turn rather than send on the
    refused route."""
    entry = getattr(agent, "_route_fallback_entry", None)
    if not isinstance(entry, Mapping) or not _gated(entry):
        return False
    if not fallback_custody_refuses(agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""), phase="turn"):
        return False
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
    return True


def check_out_of_turn_route(agent: Any) -> None:
    """F3 before a compaction outside a turn sends the transcript on the agent's own route: a kept
    fallback or a resolution-time entry is checked with ``phase="compaction"``, the session's
    labels and the transcript's data use. Nothing is restored outside a turn: a refusal raises
    :class:`RouteCustodyRefused` and the compaction does not run."""
    if getattr(agent, "_provider_fallback_active", False):
        entry = getattr(agent, "_route_fallback_entry", None)
        entry = entry if isinstance(entry, Mapping) and _gated(entry) else None
    else:
        entry = _resolution_fallback_entry(agent)
    if entry is None:
        return
    if fallback_custody_refuses(agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""),
                                phase="compaction"):
        raise RouteCustodyRefused(
            f"route custody refused {entry.get('provider')}/{entry.get('model')} for a compaction outside a turn")


@contextmanager
def resolving_session(session_id: Any) -> Iterator[None]:
    """Name the session a runtime resolution serves, so its resolve-phase custody check reads the
    labels earlier turns gave that session. Bound by the caller for that resolution only."""
    token = _RESOLVING_SESSION.set(str(session_id or ""))
    try:
        yield
    finally:
        _RESOLVING_SESSION.reset(token)


def resolution_refuses(entry: Mapping[str, Any], *, primary: dict[str, str], chain: Iterable[Mapping[str, Any]],
                       labels_cache: dict[str, Any], session_id: str = "") -> bool:
    """F2 and F3 for a resolution-time walker, which runs before any agent or turn exists: the
    custody gate sees no messages and no data use, so it decides from labels: the resolve-phase
    labels plus, when the caller names the session (*session_id* or :func:`resolving_session`),
    the labels earlier turns gave that session in this process. The thread's bound session is not
    used: a resolving thread may still carry another session's binding. The agent built on the
    chosen entry is checked again at each turn start with the turn's messages and the session's
    labels."""
    if not (_gated(entry) or _has_hook(TURN_ROUTE_HOOK)):
        return False
    session_id = str(session_id or _RESOLVING_SESSION.get() or "")
    if "labels" not in labels_cache:
        prior = session_labels(session_id)
        labels, known = announce_route(phase="resolve", route=primary, chain=_chain_routes(chain), labels=prior,
                                       session_id=session_id, reason="auth")
        labels_cache["labels"], labels_cache["known"] = prior | labels, known
    return custody_refuses(entry, phase="resolve", current=primary, labels=labels_cache["labels"],
                           session_id=session_id, labels_known=labels_cache.get("known", True))


def auxiliary_refuses(entry: Mapping[str, Any], *, task: Optional[str], failed_provider: str,
                      failed_model: Optional[str] = None) -> bool:
    """F3 for an auxiliary call (compression, titles, vision, ...) walking the main or a task
    fallback chain. The gate sees the task name, the data use stamped on the request's own
    messages, and, when the call serves an agent's turn (bound by :func:`begin_turn`, carried
    into compression workers through ContextVars), that turn's data use, session and labels. A
    required entry is skipped while no gate is loaded."""
    if not _gated(entry):
        return False
    from agent.message_metadata import data_use_requirement
    data_use = data_use_requirement(_AUX_REQUEST.get() or ())
    agent = _scope_agent()
    context: dict[str, Any] = {"session_id": "", "turn_id": "", "platform": ""}
    labels, known = frozenset(), True
    if agent is not None:
        data_use |= _turn_data_use(agent, _turn_messages(agent))
        labels, known, context = _session_labels(agent), _labels_known(agent), _agent_context(agent)
    return custody_refuses(entry, phase="auxiliary", current=_route(failed_provider, failed_model),
                           data_use=data_use, labels=labels, labels_known=known, task=str(task or ""), **context)


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
