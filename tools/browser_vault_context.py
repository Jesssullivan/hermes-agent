"""Run a vault fill inside the selected backend's ``fill_context`` (a plugin backend's lock/snapshot).

A plugin backend's failure arrives as ``PluginBackendError`` (fixed text, see
``agent/vault_backends/plugin.py``) and becomes a typed refusal; built-in backends behave as before.
"""
import json
from contextlib import nullcontext
from functools import wraps


def with_backend_fill_context(fill):
    @wraps(fill)
    def wrapped(handle, task_id=None):
        from agent.vault_backends import backend_for_handle
        from agent.vault_backends.plugin import PluginBackendError

        backend = backend_for_handle(handle)
        context = backend.fill_context() if backend is not None and hasattr(backend, "fill_context") else nullcontext()
        try:
            with context:
                return fill(handle, task_id, _backend=backend)
        except PluginBackendError as exc:
            return json.dumps({"success": False, "error_type": "backend_unavailable", "error": str(exc)})
    return wrapped
