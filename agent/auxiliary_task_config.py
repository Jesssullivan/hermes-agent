"""Read-time resolution of an ``auxiliary.<task>`` config block.

Built-in slot defaults come from ``DEFAULT_CONFIG``; plugin-registered tasks
(``PluginContext.register_auxiliary_task``) layer their declared ``defaults``
under the user's block and may inherit another slot via ``inherit_from``.
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from typing import Any, Dict

logger = logging.getLogger(__name__)

# A gateway session's frozen effective ``auxiliary:`` section (recovered secrets applied, a task whose
# secret changed returned to the primary route). Set for the turn; never the live config.yaml.
_SESSION_AUXILIARY: ContextVar = ContextVar("session_auxiliary", default=None)


@contextlib.contextmanager
def session_auxiliary_scope(auxiliary):
    token = _SESSION_AUXILIARY.set(auxiliary if isinstance(auxiliary, dict) else {})
    try:
        yield
    finally:
        _SESSION_AUXILIARY.reset(token)


class AuthoritativeAuxiliaryTask(str):
    """A task name that keeps the task's non-route policy (timeout, concurrency, extra_body, fallback
    chain) but strips its route fields, so a one-shot fallback naming a complete route never inherits
    an unset endpoint/key/wire mode from the route that just failed (#113322, #130895)."""

    __slots__ = ()
    route_authoritative = True


def authoritative_auxiliary_task(task: str | None) -> str:
    return task if isinstance(task, AuthoritativeAuxiliaryTask) else AuthoritativeAuxiliaryTask(str(task or ""))


def _get_auxiliary_task_config(task: str, _seen: frozenset = frozenset()) -> dict[str, Any]:
    """Config dict for auxiliary.<task>, or {} when unavailable. Plugin-registered tasks get their
    declared defaults layered under user config (user wins); built-in defaults live in DEFAULT_CONFIG.
    A task registered with ``inherit_from`` is resolved here, at read time, over the base task's
    effective config, so it follows the base's current settings (and the active profile's config)
    until the user pins a route on the task itself. ``_seen`` guards re-registration cycles."""
    if not task:
        return {}
    config = _task_config(str(task), _seen)
    if getattr(task, "route_authoritative", False):
        return {key: value for key, value in config.items() if key not in _AUX_ROUTE_KEYS}
    return config


def _auxiliary_section() -> dict[str, Any]:
    """The session's frozen ``auxiliary:`` section, else the active profile's config.yaml one."""
    aux = _SESSION_AUXILIARY.get()
    if aux is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except ImportError:
            return {}
        aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    return aux if isinstance(aux, dict) else {}


def main_model_fallback_enabled() -> bool:
    """``auxiliary.main_model_fallback`` (default true), read without logging."""
    from utils import is_truthy_value

    return is_truthy_value(_auxiliary_section().get("main_model_fallback"), default=True)


def main_model_fallback_refused(task: str | None, reason: str, failed_provider: str) -> bool:
    """True when ``auxiliary.main_model_fallback`` is false (default true).

    The main-agent model is the last-resort net for an explicit-provider aux slot whose own
    ``fallback_chain`` is exhausted. A deployment that must keep side traffic off the main
    model's quota (an on-prem slot that answers capacity errors while its GPU is busy) turns
    the net off for every explicit slot: the call then fails closed with its primary error, and
    no request reaches the main model. A slot on ``provider: auto`` is not covered: auto means
    the main model by design."""
    if main_model_fallback_enabled():
        return False
    logger.warning(
        "Auxiliary %s: %s on %s; main-model fallback is off (auxiliary.main_model_fallback: false), "
        "so no request goes to the main agent model", task or "call", reason, failed_provider)
    return True


def refuse_auto_rescue_off_main(task: str | None, provider: str) -> None:
    """Raise ``AuxiliaryClientUnavailable`` when ``auxiliary.main_model_fallback`` is false.

    An explicit slot that cannot build a client (bare ``openrouter`` / ``custom`` without
    credentials, a hosted vision provider whose key is missing) is otherwise re-resolved on the
    auto chain or the auto vision route, and both start at the main provider + model."""
    if (provider or "auto").strip().lower() == "auto" or main_model_fallback_enabled():
        return
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable

    logger.warning(
        "Auxiliary %s: provider %s unavailable; the auto-detection rescue is off "
        "(auxiliary.main_model_fallback: false), so no request goes to the main agent model",
        task or "call", provider)
    raise AuxiliaryClientUnavailable(
        f"Auxiliary {task or 'call'}: provider {provider} unavailable and "
        "auxiliary.main_model_fallback is false (no auto-detection rescue)")


def exhausted_layers(is_auto: bool) -> str:
    """The fallback layers an exhausted explicit or auto call walked, for its final WARNING."""
    if is_auto or main_model_fallback_enabled():
        return "fallback_chain + main agent model"
    return "fallback_chain; main-model fallback is off"


def _task_config(task: str, _seen: frozenset) -> dict[str, Any]:
    aux = _auxiliary_section()
    task_config = aux.get(task, {})
    if not isinstance(task_config, dict):
        task_config = {}
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            if _entry.get("key") == task:
                _defaults = _entry.get("defaults") or {}
                if isinstance(_defaults, dict):
                    _inherit = _entry.get("inherit_from")
                    if _inherit and task in _seen:
                        logger.warning("Auxiliary task %r has a circular inherit_from chain — "
                                       "ignoring inheritance", task)
                    if not _inherit or task in _seen:
                        return {**_defaults, **task_config}
                    base = _get_auxiliary_task_config(_inherit, _seen | {task})
                    return _layer_over_inherited({**base, **_defaults}, task_config)
                break
    except Exception:  # health: allow BLE001 -- plugin discovery must never break aux config reads
        logger.debug("plugin auxiliary task lookup failed for %r", task, exc_info=True)
    return task_config


# The fields that together pick WHERE a call goes. They travel as one unit: a provider pinned on an
# inheriting task must never pick up the base's base_url/api_key (that would send one vendor's key
# to another's endpoint).
_AUX_ROUTE_KEYS = frozenset({"provider", "model", "base_url", "api_key", "api_mode", "key_env",
                             "api_key_env", "reasoning_effort"})


def _layer_over_inherited(inherited: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Merge a task's own ``auxiliary.<task>`` block over its inherited base.

    The picker, "reset to auto" and the dashboard persist ``provider: auto`` plus ``""`` for
    model/base_url/api_key/reasoning_effort when the operator expresses no preference, so those
    placeholders mean "follow the base", not "override it with nothing". Once the operator pins a
    route (a non-auto provider, a model or a base_url) the whole route comes from the task's own
    block. Non-route keys (timeout, extra_body, ...) override per key; only ``""`` is dropped,
    never other falsy values (``reasoning_effort: false`` is an explicit choice)."""
    provider = str(user.get("provider") or "").strip().lower()
    pinned = (provider not in ("", "auto") or bool(str(user.get("model") or "").strip())
              or bool(str(user.get("base_url") or "").strip()))
    merged = {k: v for k, v in inherited.items() if not (pinned and k in _AUX_ROUTE_KEYS)}
    for key, value in user.items():
        if value == "" and not (pinned and key in _AUX_ROUTE_KEYS):
            continue
        if key == "provider" and not pinned:
            continue
        merged[key] = value
    return merged
