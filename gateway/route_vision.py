"""Route custody for gateway vision enrichment (agent/route_hooks.py).

Inbound images are described by ``vision_analyze`` before the turn's agent exists. An ``auto``
vision slot sends them on the session's resolved main route, which may be a fallback entry a
resolution-time walker chose while the resolve-phase gate saw no messages. The entry is checked
here, with the session's labels and the transcript's data use, before any image leaves."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Mapping, Optional

logger = logging.getLogger(__name__)


def _not_analyzed(message_text: str, image_paths: list[str]) -> str:
    notes = [f"[The user sent an image that was not analyzed: no model route is cleared to carry it. "
             f"Its local path is {path}]" for path in image_paths]
    prefix = "\n\n".join(notes)
    return f"{prefix}\n\n{message_text}" if message_text else prefix


async def vision_under_custody(
    enrich: Callable[[], Awaitable[str]], record: Optional[Mapping[str, Any]], runtime: Optional[Mapping[str, Any]],
    *, session_id: Any, messages: list, message_text: str, image_paths: list[str], platform: str = "",
) -> str:
    """Run *enrich* on *runtime* once custody admits the resolution-time *record* for this
    session's images. With no route hook loaded and no custody key, *enrich* runs unchanged.
    A refusal sends nothing and leaves a note with each image's path."""
    from agent.route_hooks import RouteCustodyRefused, custody_applies, detached_auxiliary, vision_rides_main_route
    if not custody_applies(record):
        return await enrich()
    from agent.auxiliary_client import scoped_runtime_main
    runtime = dict(runtime or {})
    session_id = session_id() if callable(session_id) else session_id  # looked up only when custody applies
    try:
        with scoped_runtime_main(runtime or None), detached_auxiliary(
                record, task="vision", session_id=session_id, provider=runtime.get("provider"),
                model=runtime.get("model"), messages=messages, rides_main=vision_rides_main_route(),
                platform=platform):
            return await enrich()
    except RouteCustodyRefused as exc:
        logger.warning("Vision enrichment skipped, no image sent: %s", exc)
        return _not_analyzed(message_text, image_paths)
