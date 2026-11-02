"""Route hooks for an operator's route gate: turn labels, a custody skip at failover, and the
routes-exhausted reply.

Three plugin hooks let a deployment decide which model route may carry a turn. None of them
changes anything until a plugin registers for it, or a fallback entry or the config's ``model``
block sets ``route_custody: required``.

- ``turn_route`` (F2) runs once per turn before turn-start compaction and the first API call
  (``phase="turn"``), before each later API request of the turn (``phase="request"``), before vision
  enrichment on the main route ahead of the turn (``phase="turn"``, ``reason="vision"``), and when a
  resolution-time walker is about to pick a fallback for a primary whose credentials failed
  (``phase="resolve"``). A callback may return
  ``{"labels": [...]}``. Labels are plugin-defined strings, sticky for the session within this
  process, and handed to ``route_custody``. Hermes never interprets them. A compaction that
  rotates the session id carries the labels and the data use to the child. The hook fails
  closed: when a callback raises or times out, the turn's labels are unknown, and every
  candidate a loaded ``route_custody`` gate would judge is refused for that turn.
- ``route_custody`` (F3) runs for every fallback candidate that passed the existing skips in
  ``_should_skip_fallback_candidate`` (``phase="failover"``), for every resolution-time
  candidate (``phase="resolve"``: gateway, one-shot, TUI, cron, the interactive CLI and the
  agent's init-time walker), for every main-chain or task-chain candidate of an auxiliary call
  (``phase="auxiliary"``), and at turn start (``phase="turn"``) for the serving route: the main
  route, a fallback kept from an earlier turn, or the entry a resolution-time walker built it on.
  Each walker records that entry on the agent (:func:`bind_resolution_entry`); the turn-start
  check reads the record and never infers it from the agent's own chain. A side agent, a
  cache-parity fork or a side question on such a route carries the record too
  (:func:`inherit_route_custody`, :func:`detached_auxiliary`), with the session's labels and data
  use. A delegated child on such a route carries it too (:func:`inherit_child_route_custody`), and
  whatever data use the child gathered reaches the parent (:func:`carry_child_data_use`). A model
  change on the entry's provider stays under the entry, and the gate sees the model in use.
  The serving route (F12) is judged at turn start and before every API request of a turn, a retry,
  a post-tool compaction and the budget summary included (``phase="request"``,
  :func:`check_request_route`), whether it is the main route, a kept fallback or such an entry. A
  refusal moves the turn to the first admitted route or stops it before anything is sent. The
  ``model`` block's ``route_custody: required`` marks the main route as a chain entry's key marks
  the entry: no gate, a gate error or no allow means that route carries nothing, in or out of a
  turn. A compaction outside a turn on such a route is checked first (``phase="compaction"``) and
  binds its session for the auxiliary gate (:func:`enter_compaction_scope`,
  :func:`detached_compaction`). Vision enrichment
  before a turn (gateway, ``/background``, the CLI's text image mode) checks such a route with
  ``phase="auxiliary"`` and ``task="vision"`` before an image leaves, after ``turn_route`` has
  labelled the inbound message (:func:`detached_auxiliary`, :func:`agent_vision_scope`).
  The last-resort hop to the main model asks the gate as well (:func:`main_route_refuses`), and a
  refusal in an auxiliary call ends that call's walk before built-in discovery
  (:func:`discovery_refused`), whose candidates the gate sees only when that mark covers them
  (:func:`discovery_mark_refuses`). An availability probe builds stubs that cannot send, so it
  asks no gate. :data:`HERMES_ROUTE_CUSTODY_PHASES` lists every phase.
  ``{"action": "skip"}`` refuses the candidate for this turn only. The hook fails closed: a
  callback that raises or times out refuses the candidate. An entry with ``route_custody:
  required`` also needs an explicit ``{"action": "allow"}``; any other value that is not an
  off value counts as required. While no gate is loaded such an entry carries no request.
- ``on_routes_exhausted`` (F6) runs when a turn ends on a provider error with no fallback left,
  or when custody refuses the turn's only route. It is the event a gate pings from. A callback
  may return ``{"reply": "..."}`` (or a string) to replace the turn's user-facing copy for a
  capacity ending (rate limit, overload, outage, timeout) or a custody refusal.

``HERMES_ROUTE_CAPABILITIES`` names what this build provides, keyed by the route-gate ids. F12 means
the serving route is judged as above. F1 is not listed. The ``auxiliary.main_model_fallback`` and
``compression.main_model_fallback`` keys cover explicit auxiliary slots only: a slot left on ``provider: auto`` still walks the main
model and the main fallback chain.
"""

from __future__ import annotations

import logging
import os
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
    "F12": ROUTE_CUSTODY_HOOK,
})
# Every ``phase`` a ``route_custody`` callback may receive from this build. A gate that pins a
# revision checks that each phase it handles is listed here, and refuses a phase it does not know.
HERMES_ROUTE_CUSTODY_PHASES: frozenset[str] = frozenset({"failover", "resolve", "turn", "request", "auxiliary",
                                                        "compaction"})

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
# The third slot is the session's resolution-time entry record, so a rebuilt agent on that entry's
# provider stays under it (:func:`bind_resolution_entry`).
_SESSIONS: OrderedDict[str, tuple[frozenset[str], frozenset[str], Optional[dict[str, Any]]]] = OrderedDict()
_SESSIONS_LOCK = threading.Lock()
_warned_values: set[str] = set()
# Last ``model`` block that read cleanly, by config path: (provider, marked). A later failed read
# keeps it, and a path with no good read counts as marked (fail closed).
_MAIN_MARKERS: dict[str, tuple[str, bool]] = {}
# The agent whose turn this context serves. ``begin_turn`` binds it; a worker started through
# ``propagate_context_to_thread`` (the compression pool) copies it, a thread-local would not.
_ROUTE_SCOPE: ContextVar[Optional[Callable[[], Any]]] = ContextVar("hermes_route_scope", default=None)
# The messages of the auxiliary request in flight, for the data use its custody check reports.
_AUX_REQUEST: ContextVar[Optional[list]] = ContextVar("hermes_route_aux_request", default=None)
# The refusals of the auxiliary fallback walk in flight. Built-in discovery reads it and stands down.
_AUX_WALK: ContextVar[Optional[list]] = ContextVar("hermes_route_aux_walk", default=None)
# The messages a compaction summarizes. Out of a turn they are the only record of the session's data.
_COMPACTION_SOURCE: ContextVar[Optional[list]] = ContextVar("hermes_route_compaction_source", default=None)
# The entry record of the latest runtime resolution in this context. The gateway resolves in each
# turn's own copied context, so concurrent sessions never read each other's record.
_RESOLVED_ENTRY: ContextVar[Optional[Mapping[str, Any]]] = ContextVar("hermes_route_resolved_entry", default=None)
# The session a caller is resolving a runtime for, bound only for that resolution (:func:`resolving_session`).
_RESOLVING_SESSION: ContextVar[str] = ContextVar("hermes_route_resolving_session", default="")
# False while a pre-turn vision call runs after ``turn_route`` failed on its inbound message, so its
# auxiliary fallback walk refuses every judged candidate as a turn with unknown labels does.
_INBOUND_LABELS_KNOWN: ContextVar[bool] = ContextVar("hermes_route_inbound_labels_known", default=True)


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


def _custody_value_required(raw: Any, provider: Any, model: Any) -> bool:
    """``required`` needs an allow; an off value or a missing key does not; any other value does,
    with one warning per value (fail closed)."""
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
                       raw, provider, model)
    return True


def _main_route_marker() -> tuple[str, bool]:
    """``(provider, marked)`` of the config's ``model`` block. A read that fails keeps the last good
    answer for that config, and with none the main route counts as marked."""
    try:
        from hermes_cli.config import get_config_path, read_raw_config_readonly
        from hermes_cli.config_read_errors import FailedConfigRead
        path, config = str(get_config_path()), read_raw_config_readonly()
    except Exception:  # health: allow BLE001 -- an unreadable config keeps the last good answer
        logger.debug("Route custody: config read failed", exc_info=True)
        path, config = "", None
    if config is None or isinstance(config, FailedConfigRead):
        return _MAIN_MARKERS.get(path, ("", True))
    block = config.get("model")
    if not isinstance(block, Mapping):
        _MAIN_MARKERS[path] = ("", False)
    else:
        marked = _custody_value_required(block.get(ROUTE_CUSTODY_KEY), block.get("provider"), block.get("default"))
        _MAIN_MARKERS[path] = (str(block.get("provider") or "").strip(), marked)
    return _MAIN_MARKERS[path]


def main_route_marked(*providers: Any) -> bool:
    """Whether ``model.route_custody`` covers a route on any of *providers*. A block with no provider
    or ``auto`` names no route to tell apart, so it covers every route. So does a route with no
    provider name, which proves it is not the marked one."""
    main_provider, marked = _main_route_marker()
    if not marked:
        return False
    named = [provider for provider in providers if str(provider or "").strip()]
    if not named or main_provider.lower() in ("", *_MAIN_SLOT_PROVIDERS):
        return True
    forms = _marked_forms(main_provider)
    return any(forms & _provider_forms(provider) for provider in named)


def _marked_forms(main_provider: str) -> frozenset[str]:
    """Every name a marked main route answers to. A named custom provider runs as ``custom``. A
    lookup that fails counts as named, which widens the mark. The lookup reads no credential."""
    named = main_provider.lower().startswith("custom:")
    if not named:
        try:
            from hermes_cli.runtime_provider import _get_named_custom_provider
            named = bool(_get_named_custom_provider(main_provider, resolve_credentials=False))
        except Exception:  # health: allow BLE001 -- an unreadable lookup keeps the wider mark
            logger.debug("Route custody: named custom lookup failed", exc_info=True)
            named = True
    forms = _provider_forms(main_provider)
    return forms | {"custom"} if named else forms


def entry_requires_custody(entry: Mapping[str, Any]) -> bool:
    """Whether an entry needs an explicit ``allow``: its own key does, or its provider is the main
    route's and ``model.route_custody`` marks that route."""
    return (_custody_value_required(entry.get(ROUTE_CUSTODY_KEY), entry.get("provider"), entry.get("model"))
            or main_route_marked(entry.get("provider"), entry.get("resolved_provider")))


def _gated(entry: Mapping[str, Any]) -> bool:
    """Whether any custody decision applies to *entry*: a loaded gate or a custody key."""
    return _has_hook(ROUTE_CUSTODY_HOOK) or entry_requires_custody(entry)


def _hooks_loaded() -> bool:
    return _has_hook(TURN_ROUTE_HOOK) or _has_hook(ROUTE_CUSTODY_HOOK)


def custody_applies(record: Any) -> bool:
    """Whether a route hook is loaded or the resolution-time *record* needs custody. ``False`` means
    every path behaves as if this module did not exist."""
    return _hooks_loaded() or (isinstance(record, Mapping) and entry_requires_custody(record))


def agent_custody_applies(agent: Any) -> bool:
    """:func:`custody_applies` for *agent*: also true when ``model.route_custody`` covers its route."""
    return custody_applies(out_of_turn_record(agent)) or main_route_marked(getattr(agent, "provider", ""))


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


_NO_SESSION: tuple[frozenset[str], frozenset[str], None] = (frozenset(), frozenset(), None)


def _session_state(session_id: Any) -> tuple[frozenset[str], frozenset[str]]:
    if not session_id:
        return frozenset(), frozenset()
    with _SESSIONS_LOCK:
        return _SESSIONS.get(str(session_id), _NO_SESSION)[:2]


def _session_entry(session_id: Any) -> Optional[dict[str, Any]]:
    """The resolution-time entry record the session last bound in this process."""
    if not session_id:
        return None
    with _SESSIONS_LOCK:
        return _SESSIONS.get(str(session_id), _NO_SESSION)[2]


def _hold_session(key: str, labels: frozenset[str], data_use: frozenset[str], entry: Optional[dict[str, Any]]) -> None:
    """Store a session's state as the newest. The caller holds :data:`_SESSIONS_LOCK`."""
    _SESSIONS[key] = (labels, data_use, entry)
    _SESSIONS.move_to_end(key)
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)


def _remember_entry(session_id: Any, record: Optional[dict[str, Any]]) -> None:
    if not session_id:
        return
    key = str(session_id)
    with _SESSIONS_LOCK:
        if record is not None or key in _SESSIONS:
            _hold_session(key, *_SESSIONS.get(key, _NO_SESSION)[:2], record)


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
        old_labels, old_data_use, entry = _SESSIONS.get(key, _NO_SESSION)
        _hold_session(key, old_labels | labels, old_data_use | data_use, entry)


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
    _remember_entry(getattr(agent, "session_id", None), _session_entry(old_session_id))


def carry_route_state(old_session_id: Any, new_session_id: Any, *, resident: Any = None) -> None:
    """A compaction with no agent (``session.mutate compress``) rotated *old_session_id* to
    *new_session_id*: carry the parent's labels and data use to the child, from the process
    registry and from the session's *resident* agent, whose held copies outlive a registry
    eviction. An in-place compaction keeps the id and carries nothing."""
    if not old_session_id or not new_session_id or str(old_session_id) == str(new_session_id):
        return
    labels, data_use = _session_state(old_session_id)
    if resident is not None:
        labels |= _held(resident, "_route_labels", old_session_id)
        data_use |= _held(resident, "_route_data_use", old_session_id)
    _remember_session(new_session_id, labels=labels, data_use=data_use)
    _remember_entry(new_session_id, _session_entry(old_session_id))


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
    token, walk_token = _AUX_REQUEST.set(messages if isinstance(messages, list) else None), _AUX_WALK.set([])
    try:
        yield
    finally:
        _AUX_WALK.reset(walk_token)
        _AUX_REQUEST.reset(token)


@contextmanager
def auxiliary_walk() -> Iterator[None]:
    """Open a fallback walk for a route resolved outside :func:`auxiliary_request`, such as a
    ``get_text_auxiliary_client`` call. An open walk stays the one in use."""
    if _AUX_WALK.get() is not None:
        yield
        return
    token = _AUX_WALK.set([])
    try:
        yield
    finally:
        _AUX_WALK.reset(token)


def discovery_refused(task: Optional[str]) -> bool:
    """Whether a gate refused a candidate in this auxiliary walk. Discovery then stops: its candidates
    have no fallback entry and no gate call, so it would send what the gate just kept off a route."""
    refused = _AUX_WALK.get()
    if not refused:
        return False
    logger.warning("Auxiliary %s: route custody refused %s; not trying built-in discovery", task or "call",
                   ", ".join(refused))
    return True


def in_turn(agent: Any) -> bool:
    """Whether this context serves a turn of *agent* that has begun and not ended."""
    turn = getattr(agent, "_route_turn", None)
    return _scope_agent() is agent and isinstance(turn, Mapping) and not turn.get("ended")


def end_turn(agent: Any) -> None:
    """Turn end. The scope stays bound, so a later auxiliary call in this context still reports
    the session; a later compaction counts as outside a turn (:func:`in_turn`)."""
    turn = getattr(agent, "_route_turn", None)
    if isinstance(turn, dict):
        turn["ended"] = True


def enter_compaction_scope(agent: Any, messages: Any) -> tuple[Any, Any]:
    """Bind *agent* as the route scope of a compaction and the *messages* it summarizes. Out of a
    turn no scope is bound (the gateway runs each compaction in a fresh copied context), so the
    auxiliary gate on any worker the compaction starts would see no session. The summary rows
    carry no ``data_use`` stamp, so the session keeps the data use of what it summarizes. Returns
    the tokens for :func:`exit_compaction_scope`; the first is ``None`` when *agent* already held
    the scope."""
    source = messages if isinstance(messages, list) else None
    if source and _hooks_loaded():
        from agent.message_metadata import data_use_requirement
        _grow_session(agent, data_use=data_use_requirement(source))
    scope_token = None if _scope_agent() is agent else _ROUTE_SCOPE.set(_scope_ref(agent))
    return scope_token, _COMPACTION_SOURCE.set(source)


def exit_compaction_scope(tokens: tuple[Any, Any]) -> None:
    scope_token, source_token = tokens
    _COMPACTION_SOURCE.reset(source_token)
    if scope_token is not None:
        _ROUTE_SCOPE.reset(scope_token)


class _DetachedRoute:
    """The route and session of a call that runs with no agent: a ``session.mutate compress``
    summary, or gateway vision enrichment before the turn's agent exists."""

    def __init__(self, *, session_id: str, provider: str, model: str, platform: str = "") -> None:
        self.session_id, self.provider, self.model, self.platform = session_id, provider, model, platform


@contextmanager
def _detached_scope(record: Any, *, session_id: Any, provider: Any, model: Any, messages: list, platform: str,
                    phase: str, task: str = "", check: bool = True, parent: Any = None) -> Iterator[None]:
    route = _DetachedRoute(session_id=str(session_id or ""), provider=str(provider or ""), model=str(model or ""),
                           platform=platform)
    bind_resolution_entry(route, record or _session_entry(session_id), source="detached")
    tokens, known_token = enter_compaction_scope(route, messages), None
    try:
        if check:
            known = check_out_of_turn_route(route, phase=phase, task=task,
                                            inbound=messages if phase == "auxiliary" else None)
            known_token = _INBOUND_LABELS_KNOWN.set(known)
            _check_parent_route(parent, phase=phase, task=task, messages=messages, labels_known=known)
        yield
    finally:
        if known_token is not None:
            _INBOUND_LABELS_KNOWN.reset(known_token)
        exit_compaction_scope(tokens)


def _check_parent_route(parent: Any, *, phase: str, task: str, messages: list, labels_known: bool) -> None:
    """F3 for the route of *parent*, the live agent a side call may fork from: it can differ from
    the runtime the call resolved, and a fork inherits it."""
    entry = _active_entry(parent) if parent is not None else None
    if entry is not None and fallback_custody_refuses(
            parent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""), phase=phase, task=task,
            inbound=messages, labels_known=labels_known):
        raise RouteCustodyRefused(f"route custody refused {entry.get('provider')}/{entry.get('model')} "
                                  f"for a {task} call outside a turn")


@contextmanager
def detached_compaction(record: Any, *, session_id: str, provider: str, model: str, messages: list,
                        platform: str = "") -> Iterator[None]:
    """Route scope and custody for a compaction that summarizes *messages* on a resolved runtime
    with no agent: the auxiliary gate sees the session's labels and the messages' data use, and a
    gated resolution-time *record* is checked first. Raises :class:`RouteCustodyRefused`."""
    with _detached_scope(record, session_id=session_id, provider=provider, model=model, messages=messages,
                         platform=platform, phase="compaction"):
        yield


@contextmanager
def detached_auxiliary(record: Any, *, task: str, session_id: Any, provider: Any, model: Any, messages: list,
                       rides_main: bool = True, platform: str = "", parent: Any = None) -> Iterator[None]:
    """Route scope and custody for an auxiliary call made outside any turn, such as gateway vision
    enrichment or a side question. *messages* is the session transcript and the inbound message.
    When the call rides the session's resolved main route (*rides_main*) and that route is a gated
    resolution-time *record*, ``turn_route`` labels *messages* first, then the entry is checked
    (``phase="auxiliary"``, *task*) with the session's labels and the transcript's data use. The
    route of a live *parent* the call may fork from is checked the same way. Raises
    :class:`RouteCustodyRefused`."""
    with _detached_scope(record, session_id=session_id, provider=provider, model=model, messages=messages,
                         platform=platform, phase="auxiliary", task=task, check=rides_main, parent=parent):
        yield


_MAIN_SLOT_PROVIDERS = frozenset({"auto", "main", "custom:main"})


def vision_rides_main_route(runtime: Optional[Mapping[str, Any]] = None) -> bool:
    """Whether the ``vision`` auxiliary slot sends images on the main route. A slot with no
    ``base_url`` does when its provider is ``auto``, ``main`` or ``custom:main``, read before
    normalization turns ``main`` into the main provider's name, or names the main route's own
    provider (*runtime*, else the bound main runtime). ``True`` when the slot cannot be read
    (fail closed)."""
    try:
        from agent.auxiliary_client import _normalize_vision_provider, _read_main_provider, _resolve_task_provider_model
        requested, _model, base_url, _key, _mode = _resolve_task_provider_model("vision", None, None, None, None)
        if base_url:
            return False
        raw = "".join(str(requested or "auto").split()).lower()
        if raw in _MAIN_SLOT_PROVIDERS:
            return True
        normalized = _normalize_vision_provider(raw)
        main = (runtime or {}).get("provider") or _read_main_provider()
        return normalized == "auto" or bool(_provider_forms(normalized) & _provider_forms(main))
    except Exception:  # health: allow BLE001 -- an unreadable slot is treated as the main route
        logger.debug("vision slot lookup failed; treating it as the main route", exc_info=True)
        return True


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
    return turn.get("labels_known", True) is not False and _INBOUND_LABELS_KNOWN.get()


def _agent_context(agent: Any) -> dict[str, Any]:
    turn = getattr(agent, "_route_turn", None) or {}
    return {"session_id": str(getattr(agent, "session_id", "") or ""), "turn_id": str(turn.get("turn_id") or ""),
            "platform": str(getattr(agent, "platform", "") or "")}


def _current_route(agent: Any) -> dict[str, str]:
    return _route(getattr(agent, "provider", ""), getattr(agent, "model", ""))


def fallback_custody_refuses(agent: Any, fb: Mapping[str, Any], fb_provider: str, fb_model: str, *,
                             phase: str = "failover", task: str = "", inbound: Optional[list] = None,
                             labels_known: Optional[bool] = None) -> bool:
    """F3 at failover: ``True`` skips *fb* for this turn. Not added to the unavailable set: the
    next turn may carry different data. A pre-turn call passes its *inbound* messages and the
    *labels_known* of its own ``turn_route`` call, and has no turn id."""
    if not _gated(fb):
        return False
    messages = _turn_messages(agent) if inbound is None else list(inbound)
    context = _agent_context(agent) if inbound is None else {**_agent_context(agent), "turn_id": ""}
    return custody_refuses(
        {**fb, "provider": fb_provider, "model": fb_model}, phase=phase, current=_current_route(agent),
        messages=messages, data_use=_turn_data_use(agent, messages), labels=_session_labels(agent),
        labels_known=_labels_known(agent) if labels_known is None else labels_known, task=task, **context)


def begin_turn(agent: Any, *, messages: list, turn_id: str) -> bool:
    """Turn start, before turn-start compaction and the first API call: record the turn, fire
    ``turn_route`` (F2), and judge the serving route (F3, F12). Returns ``True`` when custody moved
    the agent, so the caller publishes the new main route again. Raises
    :class:`RouteCustodyRefused` when custody refuses every route this turn could take."""
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
    return _judge_serving_route(agent, phase="turn")


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
    (``bound_route``), so a provider name the agent normalized after resolution still matches. A
    ``None`` record clears an earlier record of the same *source* (the gateway resolves again every turn),
    unless the agent still runs on that entry's provider: a gateway ``/model`` override resolves to
    ``None`` with no walk, on a cached or a rebuilt agent, so the session's record stands in. The
    record does not depend on the agent's own chain, which a safe-mode or API-locked session leaves
    empty."""
    current = getattr(agent, "_route_resolution_entry", None)
    session_id = getattr(agent, "session_id", None)
    if isinstance(record, Mapping) and (record.get("provider") or record.get("resolved_provider")):
        bound = {**record, "source": source}
        _snapshot_bound_route(agent, bound)
        agent._route_resolution_entry = bound
        if source == "resolve":
            _remember_entry(session_id, bound)
        return
    held = current if isinstance(current, Mapping) and current.get("source") == source else None
    if held is None and source == "resolve":
        held = _session_entry(session_id)
    if held is None:
        return
    if _record_provider_forms(held) & _provider_forms(_bound_route(agent)["provider"]):
        agent._route_resolution_entry = held
        return
    if held is current:
        agent._route_resolution_entry = None
    if source == "resolve":
        _remember_entry(session_id, None)


def _snapshot_bound_route(agent: Any, record: dict[str, Any]) -> None:
    """Keep the agent's route as ``bound_route`` when its provider is the entry's, so the name the
    agent uses for that provider still matches. A route on another provider is not the entry."""
    bound = _bound_route(agent)
    entry_forms = _provider_forms(record.get("provider")) | _provider_forms(record.get("resolved_provider"))
    if entry_forms & _provider_forms(bound["provider"]):
        record["bound_route"] = bound


def settle_init_resolution_route(agent: Any) -> None:
    """Agent construction finished: the init-time record takes the route the agent will send on,
    once. A later switch to another provider leaves it alone and so reads as a move off the entry."""
    record = getattr(agent, "_route_resolution_entry", None)
    if isinstance(record, dict) and record.get("source") == "init" and not record.get("settled"):
        _snapshot_bound_route(agent, record)
        record["settled"] = True


def side_route_record(parent: Any) -> Optional[dict[str, Any]]:
    """The entry record of *parent*'s current route, for an agent built by copying that route: its
    kept fallback, or the resolution-time entry it still runs on."""
    if getattr(parent, "_provider_fallback_active", False):
        entry = getattr(parent, "_route_fallback_entry", None)
        if not isinstance(entry, Mapping):
            return None
        return resolution_record(entry, {"provider": getattr(parent, "provider", "")}, model=getattr(parent, "model", ""))
    entry = _resolution_fallback_entry(parent)
    return dict(entry) if entry is not None else None


def inherit_route_custody(child: Any, record: Any, *, parent: Any = None, session_id: Any = None) -> None:
    """Custody for *child*, an agent built for a session with its own session id or as a fork:
    bind the entry *record* of the route it was built on, so its turn start checks it, and seed the
    labels and data use of the session it serves (*session_id*, else *parent*'s), including
    *parent*'s held copies. A side agent's own session has neither, and its gate must not read as
    clean. A named *session_id* with no *record* uses the session's own entry record, as a gateway
    ``/model`` override on that entry resolves to none. With no route hook loaded and no custody key,
    it does nothing."""
    record = record or (_session_entry(session_id) if session_id else None)
    if not custody_applies(record):
        return
    if isinstance(record, Mapping):
        bind_resolution_entry(child, record, source="side")
    source_id = session_id or getattr(parent, "session_id", None)
    labels, data_use = _session_state(source_id)
    if parent is not None:
        labels |= _held(parent, "_route_labels", source_id)
        data_use |= _held(parent, "_route_data_use", source_id)
    _grow_session(child, labels=labels, data_use=data_use)


def inherit_child_route_custody(child: Any, record: Any, parent: Any) -> None:
    """Custody for a delegated *child*. *record* is :func:`side_route_record` of *parent*, read when
    the child's runtime was copied. The child binds it unless a delegation override put it on another
    provider, and it always takes the parent's labels and data use. A child's turn start then checks
    the route as any other agent's does, and a refusal fails that child alone."""
    if isinstance(record, Mapping):
        provider = _provider_forms(getattr(child, "provider", ""))
        if provider and not provider & _record_provider_forms(record):
            record = None
    inherit_route_custody(child, record, parent=parent)


def carry_child_data_use(parent: Any, child: Any, result: Any = None) -> None:
    """A delegated *child* ended, on any path: its summary reaches *parent*, so what the child read
    does too. The union goes to the parent's session, for every later gate call, and to the
    ``delegate_task`` tool row through the dispatch sink, for the requests that carry that row. A
    background child has no open sink, so only the session takes it."""
    from agent.message_metadata import data_use_requirement
    from tools.connectors.data_use import record_data_use
    rows = (getattr(child, "_session_messages", None), result.get("messages") if isinstance(result, Mapping) else None)
    data_use = _session_data_use(child).union(*(data_use_requirement(r) for r in rows if isinstance(r, list)))
    if data_use:
        _grow_session(parent, data_use=data_use)
        record_data_use(data_use)


def _record_provider_forms(record: Mapping[str, Any]) -> frozenset[str]:
    """Every provider name a resolution-time *record* answers to."""
    bound = record.get("bound_route") if isinstance(record.get("bound_route"), Mapping) else {}
    return (_provider_forms(record.get("provider")) | _provider_forms(record.get("resolved_provider"))
            | _provider_forms(bound.get("provider")))


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


def provider_matches(first: Any, second: Any) -> bool:
    """Whether two provider names reach one provider, in canonical form on both sides."""
    return bool(_provider_forms(first) & _provider_forms(second))


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
    still runs on it. The agent's provider matches the entry as configured, as resolved, or as the
    agent was built on it (``bound_route``), in canonical form on every side. Any other model on
    that provider (``/model``, a request route, ``delegation.model``, a rewrite such as
    ``nous/welcome``) is still the entry's route, with the model in use, so the gate judges what
    will carry the data. So is a blank provider, which proves no move. ``None`` with no record, no
    gate, or after the agent moved to another provider, which is logged once at warning."""
    record = getattr(agent, "_route_resolution_entry", None)
    if not isinstance(record, Mapping) or not _gated(record):
        return None
    current = _current_route(agent)
    if not current["provider"] or _record_provider_forms(record) & _provider_forms(current["provider"]):
        models = (str(record.get("model") or ""), str(record.get("resolved_model") or ""))
        if not current["model"] or _same_model(models, current["model"], current["provider"]):
            return record
        return {**record, "model": current["model"], "resolved_model": current["model"]}
    if isinstance(record, dict) and not record.get("left_warned"):
        record["left_warned"] = True
        logger.warning("Route custody: the agent left the resolution-time entry %s/%s (now %s/%s); "
                       "turn starts no longer check it", record.get("provider"), record.get("model"),
                       current["provider"], current["model"])
    return None


def _judged_route(agent: Any) -> Optional[Mapping[str, Any]]:
    """What the gate answers for on the route the next request leaves on: the gated entry, else the
    agent's own route while a gate is loaded. An unmarked main route is the one route a labeller
    failure alone leaves the turn on."""
    entry = _active_entry(agent)
    if entry is None and _has_hook(ROUTE_CUSTODY_HOOK) and _labels_known(agent):
        return _current_route(agent)
    return entry


def _judge_serving_route(agent: Any, *, phase: str) -> bool:
    """F12: judge the route the next request would leave on. ``True`` when a refusal moved the agent
    to the first admitted route. Raises :class:`RouteCustodyRefused` when none admits the turn."""
    entry = _judged_route(agent)
    if entry is None or not _serving_entry_refused(agent, entry, phase=phase):
        return False
    _move_off_refused_route(agent, entry, phase=phase)
    return True


def _serving_entry_refused(agent: Any, entry: Mapping[str, Any], *, phase: str) -> bool:
    return fallback_custody_refuses(agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""),
                                    phase=phase)


def _move_off_refused_route(agent: Any, entry: Mapping[str, Any], *, phase: str) -> None:
    """The gate refused *entry*, the serving route. A kept fallback goes back to the primary, which
    is judged in turn. Otherwise the chain is walked on, and :func:`fallback_custody_refuses` judges
    each candidate. Raises :class:`RouteCustodyRefused` before anything is sent when no route admits
    the turn. The agent keeps its route then, and the caller sends nothing on it."""
    if getattr(agent, "_provider_fallback_active", False):
        with suppress(RouteCustodyRefused):
            _leave_kept_fallback(agent, entry)
            restored = _judged_route(agent)
            if restored is None or not _serving_entry_refused(agent, restored, phase=phase):
                return
    if not _walk_fallback_chain(agent):
        raise RouteCustodyRefused(f"route custody refused {entry.get('provider')}/{entry.get('model')}, and no "
                                  "other route admits this turn")


def _walk_fallback_chain(agent: Any) -> bool:
    """Activate the next chain entry the gate admits. A custody move says nothing about the primary's
    health, so the walk leaves the primary's cooldown as it found it."""
    walk = getattr(agent, "_try_activate_fallback", None)
    if not callable(walk) or not getattr(agent, "_fallback_chain", None):
        return False
    held_until = getattr(agent, "_rate_limited_until", 0) or 0
    try:
        return walk() is True
    except Exception:  # health: allow BLE001 -- a walk that cannot finish admits nothing
        logger.warning("Route custody: the fallback walk failed; stopping the turn", exc_info=True)
        return False
    finally:
        agent._rate_limited_until = held_until


def _leave_kept_fallback(agent: Any, entry: Mapping[str, Any]) -> None:
    """Custody refused the kept fallback *entry*: back to the primary, keeping its cooldown. Raises
    :class:`RouteCustodyRefused` when the primary cannot be restored."""
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


def out_of_turn_record(agent: Any) -> Any:
    """The entry record the agent's own route came from: its kept fallback while one is active,
    else its resolution-time record."""
    if getattr(agent, "_provider_fallback_active", False):
        return getattr(agent, "_route_fallback_entry", None)
    return getattr(agent, "_route_resolution_entry", None)


def _active_entry(agent: Any) -> Optional[Mapping[str, Any]]:
    """The gated entry the agent's own route came from, or ``None``: a kept fallback while one is
    active, else a resolution-time entry the agent still runs on, else the route itself when
    ``model.route_custody`` covers it. A fallback active with no recorded entry counts as that route."""
    if getattr(agent, "_provider_fallback_active", False):
        entry = getattr(agent, "_route_fallback_entry", None)
        if isinstance(entry, Mapping):
            return entry if _gated(entry) else None
    else:
        entry = _resolution_fallback_entry(agent)
        if entry is not None:
            return entry
    route = _current_route(agent)
    return route if entry_requires_custody(route) else None


def check_request_route(agent: Any, messages: list) -> bool:
    """F12 before every API request of a turn and before a post-tool compaction. *messages* is the
    live transcript: a tool row may have stamped ``data_use`` since the last check. ``turn_route``
    labels it and the serving route is judged with ``phase="request"``, on every call. A refusal
    moves the agent to the first admitted route (``True``, so the caller publishes the new main
    route). :class:`RouteCustodyRefused` means no route admits the turn, before anything is sent.
    ``False`` means the route stands."""
    if not agent_custody_applies(agent):
        return False
    from agent.message_metadata import data_use_requirement
    turn = getattr(agent, "_route_turn", None)
    if isinstance(turn, dict):
        turn["messages"] = messages  # a compaction replaces the list; a failover must read this one
    _grow_session(agent, data_use=data_use_requirement(messages))
    if _has_hook(TURN_ROUTE_HOOK):
        labels, known = announce_route(
            phase="request", route=_current_route(agent), chain=_chain_routes(getattr(agent, "_fallback_chain", None)),
            messages=messages, data_use=_turn_data_use(agent, messages), labels=_session_labels(agent),
            reason="fallback" if getattr(agent, "_provider_fallback_active", False) else "",
            **_agent_context(agent))
        _grow_session(agent, labels=labels)
        if isinstance(turn, dict) and not known:
            turn["labels_known"] = False  # sticky for the turn, as a failure at turn start is
    return _judge_serving_route(agent, phase="request")


def check_out_of_turn_route(agent: Any, *, phase: str = "compaction", task: str = "",
                            inbound: Optional[list] = None) -> bool:
    """F3 before a call outside a turn sends the transcript on the agent's own route: a kept
    fallback or a resolution-time entry is checked with *phase* (``compaction``, or ``auxiliary``
    with *task* for vision enrichment), the session's labels and the transcript's data use. A
    pre-turn call passes its *inbound* messages (the transcript and the inbound message):
    ``turn_route`` labels them first (``phase="turn"``, *task* as its ``reason``), so labels drawn from
    the inbound message reach the check and the call's own fallback walk, and unknown labels
    refuse. Returns whether the labels are known. Nothing is restored outside a turn: a refusal
    raises :class:`RouteCustodyRefused` and the call does not run."""
    known = None
    if inbound is not None:
        labels, known = announce_route(
            phase="turn", route=_current_route(agent), chain=_chain_routes(getattr(agent, "_fallback_chain", None)),
            messages=inbound, data_use=_turn_data_use(agent, inbound), labels=_session_labels(agent),
            reason=task, **{**_agent_context(agent), "turn_id": ""})
        _grow_session(agent, labels=labels)
    entry = _active_entry(agent)
    if entry is not None and fallback_custody_refuses(
            agent, entry, str(entry.get("provider") or ""), str(entry.get("model") or ""), phase=phase, task=task,
            inbound=inbound, labels_known=known):
        raise RouteCustodyRefused(f"route custody refused {entry.get('provider')}/{entry.get('model')} for "
                                  f"{'a ' + task + ' call' if task else 'a compaction'} outside a turn")
    return known is not False


_AGENT_RUNTIME_FIELDS = ("provider", "model", "requested_provider", "base_url", "api_key", "api_mode", "auth_mode",
                         "session_id")


@contextmanager
def agent_vision_scope(agent: Any, *, messages: list) -> Iterator[None]:
    """Route custody for vision enrichment a front end runs on *agent*'s route before its turn
    (the CLI's text image mode). *messages* is the transcript and the inbound message. While a
    route hook is loaded or the agent's kept fallback or resolution-time record carries a custody
    key, the call is bound to the agent's own runtime and route scope, and a vision slot that rides
    that route is checked as :func:`detached_auxiliary` does. With neither, or with no agent, the
    call runs unchanged. Raises :class:`RouteCustodyRefused` before any image leaves."""
    if agent is None or not agent_custody_applies(agent):
        yield
        return
    from agent.auxiliary_client import scoped_runtime_main
    runtime = {field: getattr(agent, field, "") or "" for field in _AGENT_RUNTIME_FIELDS}
    with scoped_runtime_main(runtime):
        tokens, known_token = enter_compaction_scope(agent, messages), None
        try:
            if vision_rides_main_route(runtime):
                known = check_out_of_turn_route(agent, phase="auxiliary", task="vision", inbound=messages)
                known_token = _INBOUND_LABELS_KNOWN.set(known)
            yield
        finally:
            if known_token is not None:
                _INBOUND_LABELS_KNOWN.reset(known_token)
            exit_compaction_scope(tokens)


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
    required entry is skipped while no gate is loaded. An availability probe asks no gate: it builds
    stubs that cannot send, so a refusal there would only hide a tool, and its real call is judged."""
    if not _gated(entry) or _availability_probe():
        return False
    from agent.message_metadata import data_use_requirement
    data_use = data_use_requirement(_AUX_REQUEST.get() or ())
    agent = _scope_agent()
    context: dict[str, Any] = {"session_id": "", "turn_id": "", "platform": ""}
    labels, known = frozenset(), True
    if agent is not None:
        data_use |= _turn_data_use(agent, _turn_messages(agent))
        labels, known, context = _session_labels(agent), _labels_known(agent), _agent_context(agent)
    refused = custody_refuses(entry, phase="auxiliary", current=_route(failed_provider, failed_model),
                              data_use=data_use, labels=labels, labels_known=known, task=str(task or ""), **context)
    walk = _AUX_WALK.get()
    if refused and walk is not None:
        walk.append(f"{entry.get('provider')}/{entry.get('model')}")
    return refused


def main_route_refuses(provider: Any, model: Any, *, task: Optional[str], failed_provider: str,
                       failed_model: Optional[str] = None) -> bool:
    """F3 for an auxiliary call's last-resort hop to the main model. The main route has no chain entry
    of its own, so the candidate takes the custody key of the gated entry the scope agent runs on."""
    agent = _scope_agent()
    active = _active_entry(agent) if agent is not None else None
    return auxiliary_refuses({**(active or {}), "provider": provider, "model": model}, task=task,
                             failed_provider=failed_provider, failed_model=failed_model)


def marked_main_refuses(provider: Any, model: Any, *, task: Optional[str], base_url: Any = None) -> bool:
    """F3 for an auxiliary slot about to ride a route ``model.route_custody`` marks. A slot's spelling
    (``main``, ``custom:main``, ``custom:<provider>``, ``codex``) can name another route than the one
    it resolves to, so the mark is read on both, and the gate sees the first name the mark covers.
    A slot whose *base_url* reaches the marked route's endpoint is judged too (:func:`_endpoint_route`)."""
    if not _main_route_marker()[1]:
        return False
    from agent.auxiliary_client import _normalize_aux_provider
    resolved = _normalize_aux_provider(provider) if str(provider or "").strip() else ""
    route = next((name for name in (resolved, provider) if main_route_marked(name)), None)
    if route is None:
        route = _endpoint_route(resolved or provider, base_url)
    return route is not None and main_route_refuses(route, model, task=task, failed_provider=str(route or ""),
                                                    failed_model=str(model or ""))


def _endpoint_route(provider: Any, base_url: Any) -> Optional[str]:
    """The name the gate sees for a slot at *base_url* that no marked name covers: the marked provider
    when the slot's scheme, host and port are the marked route's, and the slot's own name (else
    ``custom``) when the marked route's endpoint cannot be resolved (fail closed). ``None`` otherwise."""
    url = str(base_url or "").strip()
    if not url:
        return None
    from utils import base_url_origin
    origins, origin = _marked_route_origins(), base_url_origin(url)
    if not origins:
        if "\0endpoint" not in _warned_values:
            _warned_values.add("\0endpoint")
            logger.warning("Route custody: the marked route's endpoint cannot be resolved; judging every base_url slot")
        return str(provider or "").strip() or "custom"
    if origin in origins or not origin[1]:
        return _main_route_marker()[0]
    return None


def _marked_route_origins() -> frozenset[tuple[str, str, int]]:
    """``(scheme, host, port)`` of every endpoint the marked route can reach: the block's ``base_url``,
    the named custom provider's, its provider's registry and catalog endpoints and their env
    overrides, and the bound main runtime's when the mark covers that runtime. Empty when none resolves or a
    lookup fails, so the caller judges every slot with a ``base_url``. Reads no credential."""
    from utils import base_url_origin
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.config import read_raw_config_readonly
        from hermes_cli.config_read_errors import FailedConfigRead
        from hermes_cli.runtime_provider import _get_named_custom_provider
        from agent.auxiliary_client import _normalize_main_runtime
        config = read_raw_config_readonly()
        if config is None or isinstance(config, FailedConfigRead):
            return frozenset()
        block = config.get("model") if isinstance(config.get("model"), Mapping) else {}
        main_provider = _main_route_marker()[0]
        urls = [block.get("base_url")]
        named = _get_named_custom_provider(main_provider, resolve_credentials=False) if main_provider else None
        urls.append((named or {}).get("base_url"))
        forms = _provider_forms(main_provider)
        for form in forms:
            pconfig = PROVIDER_REGISTRY.get(form)
            if pconfig is not None:
                urls += [pconfig.inference_base_url, os.environ.get(pconfig.base_url_env_var or "")]
            with suppress(Exception):  # the catalog adds endpoints; a failed lookup removes none
                from hermes_cli.providers import get_provider
                pdef = get_provider(form, allow_network=False)
                urls += [pdef.base_url, os.environ.get(pdef.base_url_env_var or "")] if pdef else []
        if "openrouter" in forms:
            from hermes_constants import OPENROUTER_BASE_URL
            urls.append(OPENROUTER_BASE_URL)
        runtime = _normalize_main_runtime(None)
        if str(runtime.get("provider") or "").strip() and main_route_marked(runtime.get("provider")):
            urls.append(runtime.get("base_url"))
    except Exception:  # health: allow BLE001 -- an unresolved endpoint judges every base_url slot
        logger.debug("Route custody: marked endpoint lookup failed", exc_info=True)
        return frozenset()
    origins = {base_url_origin(str(url)) for url in urls if isinstance(url, str) and url.strip()}
    return frozenset(origin for origin in origins if origin[1])


# Built-in text discovery labels a candidate by its chain slot. ``local/custom`` runs as ``custom``;
# ``api-key`` is whichever API-key provider answered first, so it names no one route.
_DISCOVERY_ROUTES = {"local/custom": "custom", "api-key": ""}


def discovery_mark_refuses(label: Any, model: Any, *, task: Optional[str], base_url: Any = None) -> bool:
    """F12 for the client built-in text discovery picked, judged on the model it picked and on the
    client's *base_url*. Under a marked block, a candidate that names no one provider is refused
    with no gate call."""
    if not _main_route_marker()[1] or _availability_probe():
        return False
    name = str(label or "").strip().lower()
    route = _DISCOVERY_ROUTES.get(name, name)
    if route:
        return marked_main_refuses(route, model, task=task, base_url=base_url)
    logger.warning("Auxiliary %s: model.route_custody is set and discovery picked %s, which names no one "
                   "provider; not sending", task or "call", name or "an unnamed route")
    return True


def client_base_url(client: Any) -> str:
    """The base URL a built client sends to, read from it or from the client it wraps; ``""`` if none."""
    import httpx
    for obj in (client, getattr(client, "_client", None), getattr(client, "client", None)):
        url = getattr(obj, "base_url", None) if obj is not None else None
        if isinstance(url, (str, httpx.URL)) and str(url).strip():
            return str(url)
    return ""


def judged_auto_model(label: Any, client: Any, model: Any, judged: Any, *, task: Optional[str]) -> bool:
    """F12 for an ``auto`` text slot that names its own model: the walk judged the route *label* names
    on *judged*, and *client* sends *model*. Judged again on *model* when the two differ."""
    if model == judged:
        return False
    name = str(label or "").strip().lower()
    return marked_main_refuses(_DISCOVERY_ROUTES.get(name, name), model, task=task, base_url=client_base_url(client))


def judged_vision_client(provider: Any, client: Any, model: Any, judged: Any,
                         base_url: Any = None) -> tuple[Any, Any]:
    """``(client, model)`` for a vision slot, judged again (F12) when the model it sends is not the
    *judged* one the gate already saw, and ``(None, None)`` when there is no client or a refusal."""
    if client is None or (model != judged and marked_main_refuses(provider, model, task="vision", base_url=base_url)):
        return None, None
    return client, model


def _availability_probe() -> bool:
    from agent.auxiliary_client import _aux_probe_active
    return _aux_probe_active()


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


def route_custody_stopped(agent: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Turn result when custody refused the route within a turn (:func:`check_request_route`). Rows
    already sent stay in the transcript; nothing was sent after them. Fires ``on_routes_exhausted``
    with ``reason="route_custody"``."""
    if not isinstance(result, dict):
        return result
    result.update(completed=False, partial=True, failed=True, error=result.get("final_response"),
                  turn_exit_reason=ROUTE_CUSTODY_REASON, failure_reason=ROUTE_CUSTODY_REASON, failure_retryable=False)
    if not _has_hook(ROUTES_EXHAUSTED_HOOK):
        return result
    return _fire_routes_exhausted(agent, result, reason=ROUTE_CUSTODY_REASON, retryable=False, reset_at=None,
                                  provider=getattr(agent, "provider", ""), model=getattr(agent, "model", ""))


def _reply_from(result: Any) -> str:
    if isinstance(result, Mapping):
        result = result.get("reply")
    return result.strip()[:_MAX_REPLY_CHARS] if isinstance(result, str) else ""
