"""Route cooldowns that outlive the agent (route-gate id F9, TIN-7765).

``_rate_limited_until`` lives on one ``AIAgent``. The gateway drops that agent after a successful
fallback turn, ``api_server`` builds one per run and cron builds one per job, so the next turn
starts on the walled route and pays the wall again. Each route here gets a small file under
``<HERMES_HOME>/rate_limits/``, so every session and process of a home reads the same deadline.

One file per route keeps a write from touching another route's deadline, and a lost race on one
route costs at most one wall. Deadlines are wall-clock epochs, because monotonic time does not
cross a process. A missing, corrupt or unwritable file reads as no cooldown: this store can only
save a call, never refuse a turn. The file holds no credential, and ``base_url`` reaches it only
as part of a hash.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from utils import atomic_write_text

logger = logging.getLogger(__name__)

# Longer than the weekly plan window. A deadline past this is a bad clock or a bad file, not a reset.
MAX_COOLDOWN_SECONDS = 8 * 86400


@dataclass(frozen=True)
class RouteCooldown:
    until: float  # wall-clock epoch
    count: int  # walls since this route last served a response
    reason: str
    armed_at: float

    def remaining(self) -> float:
        return max(0.0, self.until - time.time())


def _route_key(provider: Any, model: Any, base_url: Any) -> str:
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip()
    # A chain entry names its model as configured and the binder normalizes it, so both forms must
    # land on one file. A custom provider's name does not identify its endpoint, so its URL does.
    with contextlib.suppress(Exception):
        from hermes_cli.model_normalize import normalize_model_for_provider
        model = normalize_model_for_provider(model, provider) or model
    custom = provider == "custom" or provider.startswith("custom:")
    endpoint = str(base_url or "").strip().rstrip("/").lower() if custom else ""
    return hashlib.sha256("\0".join((provider, model, endpoint)).encode("utf-8")).hexdigest()[:24]


def _path(provider: Any, model: Any, base_url: Any) -> str:
    from hermes_constants import get_hermes_home
    return os.path.join(str(get_hermes_home()), "rate_limits", f"route-{_route_key(provider, model, base_url)}.json")


def _load(path: str) -> Optional[RouteCooldown]:
    try:
        with open(path, encoding="utf-8-sig") as handle:
            raw = json.load(handle)
        cooldown = RouteCooldown(
            until=float(raw["until"]), count=max(0, int(raw["count"])), reason=str(raw["reason"]),
            armed_at=float(raw["armed_at"]),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if not (math.isfinite(cooldown.until) and math.isfinite(cooldown.armed_at)):
        return None
    return None if cooldown.until - time.time() > MAX_COOLDOWN_SECONDS else cooldown


def read_route_cooldown(provider: Any, model: Any, base_url: Any = "") -> Optional[RouteCooldown]:
    """The route's record, expired or not, so a repeat wall can keep doubling; None when absent."""
    try:
        return _load(_path(provider, model, base_url))
    except Exception:
        logger.debug("Route cooldown read failed", exc_info=True)
        return None


def route_cooldown_remaining(provider: Any, model: Any, base_url: Any = "") -> float:
    cooldown = read_route_cooldown(provider, model, base_url)
    return cooldown.remaining() if cooldown is not None else 0.0


def record_route_cooldown(provider: Any, model: Any, base_url: Any, *, seconds: float, count: int, reason: str) -> None:
    """Arm the route for ``seconds``. A later deadline another session already armed stands."""
    try:
        path = _path(provider, model, base_url)
        now = time.time()
        until = now + min(max(0.0, float(seconds)), MAX_COOLDOWN_SECONDS)
        held = _load(path)
        if held is not None:
            until = max(until, held.until)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_write_text(path, json.dumps({
            "provider": str(provider or ""), "model": str(model or ""), "until": until, "count": int(count),
            "reason": str(reason), "armed_at": now,
        }))
    except Exception:
        logger.debug("Route cooldown write failed", exc_info=True)


def clear_route_cooldown(provider: Any, model: Any, base_url: Any = "", *, started_at: Optional[float] = None) -> None:
    """A served response proves the route open. A response whose request started before the wall
    proves nothing about it: a long stream can end after another session walled."""
    try:
        path = _path(provider, model, base_url)
        held = _load(path)
        if held is not None and started_at is not None and held.armed_at > started_at:
            return
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
    except Exception:
        logger.debug("Route cooldown clear failed", exc_info=True)
