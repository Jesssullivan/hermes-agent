"""Run a vault fill, or an automatic one-time-code entry, inside the selected backend's ``fill_context``
(a plugin backend's lock/snapshot/manager session).

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


def write_backend_code(backend, handle, write):
    """Mint the backend's one-time code for ``handle`` and ``write`` it, both inside ``fill_context``, as a
    password fill is. ``None`` when the backend has no code (or failed to give one): the caller asks the user."""
    from agent.vault_backends.plugin import PluginBackendError

    written = None
    try:
        with backend.fill_context():
            try:
                code = backend.resolve_otp(handle)
            except Exception:  # health: allow BLE001 -- no automatic code; the user is asked instead
                code = None
            if code:
                written = write(code)
    except PluginBackendError:  # the plugin's context failed to open or close; a code already written stands
        pass
    return written
