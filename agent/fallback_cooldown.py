"""Primary rate-limit cooldown arming and per-session model rejection markers, shared by the
fallback walk (chat_completion_helpers) and restore_primary_runtime (agent_runtime_helpers)."""
import logging
import math
import time
from typing import Any

from agent.error_classifier import FailoverReason
from agent.route_cooldown import read_route_cooldown, record_route_cooldown

logger = logging.getLogger(__name__)

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})


def _provider_reset_delay(reset_at) -> float | None:
    """Seconds until the provider-declared reset, or None when missing/invalid/expired."""
    from agent.credential_pool import _parse_absolute_timestamp
    parsed = _parse_absolute_timestamp(reset_at)
    delay = parsed - time.time() if parsed is not None else None
    if delay is not None and math.isfinite(delay) and delay > 0:
        return delay
    return None


def switch_deferred_by_reset(agent, reason: FailoverReason | None, reset_at) -> bool:
    """Opt-in ``fallback.min_switch_reset_seconds`` (default 0 = off, #117484): when the primary's
    rate limit reopens sooner than N seconds, switching model mid-task costs more than waiting, so
    the fallback walk is skipped and the retry loop's own backoff rides out the window. Only for
    rate-limit failovers leaving the primary with a valid future ``reset_at``."""
    if reason not in _RATE_LIMIT_FAILOVER_REASONS or getattr(agent, "_fallback_activated", False):
        return False
    try:
        from hermes_cli.config import load_config
        threshold = float((load_config() or {}).get("fallback", {}).get("min_switch_reset_seconds") or 0)
    except Exception:
        return False
    if threshold <= 0:
        return False
    delay = _provider_reset_delay(reset_at)
    if delay is None or delay >= threshold:
        return False
    logging.info("Rate limit resets in %.0f s (< fallback.min_switch_reset_seconds=%.0f): staying on the primary", delay, threshold)
    return True


def _arm_rate_limit_cooldown(
    agent, reason: FailoverReason | None, reset_at=None,
) -> int | None:
    """Arm the primary cooldown until the provider reset, or use exponential backoff.

    ``reset_at`` is an absolute wall-clock timestamp while ``_rate_limited_until`` is monotonic;
    convert through a duration so wall-clock epoch values never enter the monotonic comparison.
    Missing, invalid, or expired provider resets retain the 60s → 2m → ... → 4h fallback.
    The failing route goes to the shared store (agent/route_cooldown.py) on every hop, so its window
    outlives this agent and a fallback's wall is recorded too. ``_rate_limited_until`` stays the
    primary's: chain-switching from an active fallback means the primary was not the failing source.
    The doubling reads the store as well, because a fresh agent starts at zero. Return the armed
    primary cooldown in seconds, or None when not armed.
    """
    if reason not in _RATE_LIMIT_FAILOVER_REASONS:
        return None
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
    on_primary = not getattr(agent, "_fallback_activated", False) or bool(primary_provider and current_provider == primary_provider)
    route = (getattr(agent, "provider", ""), getattr(agent, "model", ""), getattr(agent, "base_url", ""))
    shared = read_route_cooldown(*route)
    backoff_count = max(getattr(agent, "_rate_limit_backoff_count", 0) if on_primary else 0, shared.count if shared else 0)
    provider_delay = _provider_reset_delay(reset_at)
    if provider_delay is not None:
        backoff_seconds = math.ceil(provider_delay)
        source = "provider reset"
    else:
        backoff_seconds = min(60 * (2 ** backoff_count), 14400)
        source = "exponential fallback"
    record_route_cooldown(*route, seconds=backoff_seconds, count=backoff_count + 1, reason=reason.value)
    if not on_primary:
        return None
    agent._rate_limit_backoff_count = backoff_count + 1
    agent._rate_limited_until = time.monotonic() + backoff_seconds
    logging.info(
        "Rate-limit backoff level %d: cooldown %d s (%.1f min, backoff#%d, %s)",
        backoff_count, backoff_seconds, backoff_seconds / 60, backoff_count + 1, source,
    )
    return backoff_seconds


def _candidate_pool_exhausted(agent, fb_provider: str, fb_model: str) -> bool:
    """True when every credential the candidate would use sits in an exhaustion cooldown longer
    than the retry loop's longest wait (the 600s Retry-After cap): switching to it only fails the
    turn the same way the primary just did (#89401). A short throttle still gets its chance."""
    pool = getattr(agent, "_credential_pool", None)
    if pool is None or (getattr(pool, "provider", "") or "").strip().lower() != fb_provider:
        try:
            from agent.credential_pool import load_pool
            pool = load_pool(fb_provider)
        except Exception:
            return False
    if pool is None or not pool.has_credentials() or pool.has_available(model=fb_model):
        return False
    until = pool.next_available_at(model=fb_model)
    return until is None or until - time.time() > 600


def hold_walled_primary(agent: Any) -> bool:
    """Turn start, after custody knows this turn's labels: leave a primary whose shared cooldown is
    still armed, so the turn walks the chain without paying the wall. ``True`` when the agent moved.
    The walk applies every skip a mid-turn fallback applies, custody included. When no other route
    will take the turn the primary keeps it, and its own failure path decides how the turn ends."""
    if getattr(agent, "_fallback_activated", False) or not getattr(agent, "_fallback_chain", None):
        return False
    shared = read_route_cooldown(getattr(agent, "provider", ""), getattr(agent, "model", ""), getattr(agent, "base_url", ""))
    remaining = shared.remaining() if shared is not None else 0.0
    if remaining <= 0:
        return False
    try:
        reason = FailoverReason(shared.reason)
    except ValueError:
        reason = FailoverReason.rate_limit
    held_until = getattr(agent, "_rate_limited_until", 0) or 0
    agent._rate_limited_until = max(held_until, time.monotonic() + remaining)
    if agent._try_activate_fallback(reason, held_seconds=math.ceil(remaining)):
        return True
    agent._rate_limited_until = held_until
    agent._fallback_index = 0  # the walk found nothing; the primary's own failure walks again
    return False


def _mark_entitlement_rejected_model(agent, api_error) -> bool:
    """Record a Codex ChatGPT-account 400 that rejects the current model for this account.

    Pool rotation runs first (recover_with_credential_pool benches (credential, model) and
    moves to the next entitled entry, #71970); this runs only once no pool entry is left for
    the model, so the (provider, model) pair is treated as dead for the session: the fallback
    walk skips it and restore_primary_runtime stops switching back — otherwise every turn
    re-fails on the primary, announces an unverified "Primary model restored", and oscillates
    forever (#106475).
    """
    if getattr(api_error, "status_code", None) != 400:
        return False
    from agent.error_classifier import CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER
    haystack = str(getattr(api_error, "message", "") or api_error).lower()
    if CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER not in haystack:
        return False
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    model = str(getattr(agent, "model", "") or "").strip()
    if not provider or not model:
        return False
    pool = getattr(agent, "_credential_pool", None)
    if pool is not None and pool.has_available(model=model):
        return False  # another pool entry is still eligible for this model; rotation owns it
    rejected = getattr(agent, "_entitlement_rejected_models", None)
    if rejected is None:
        rejected = agent._entitlement_rejected_models = set()
    if (provider, model) in rejected:
        return True
    rejected.add((provider, model))
    logger.warning(
        "Model entitlement rejection: this account is not entitled to %s via %s; "
        "treating it as unavailable for this session",
        model, provider,
    )
    agent._buffer_diagnostic_status(
        f"🚫 This account is not entitled to {model} via {provider}; it will be skipped "
        "until restart. Switch to an entitled model via /model or `hermes model`."
    )
    return True


def _is_entitlement_rejected(agent, provider: str, model: str) -> bool:
    """True when (provider, model) — as configured or normalized — was rejected as unentitled
    for this account (see _mark_entitlement_rejected_model)."""
    rejected = getattr(agent, "_entitlement_rejected_models", None) or ()
    if not rejected:
        return False
    if (provider, model) in rejected:
        return True
    from hermes_cli.model_normalize import normalize_model_for_provider
    return (provider, normalize_model_for_provider(model, provider)) in rejected
