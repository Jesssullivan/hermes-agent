"""Core-owned wrapper around a plugin-registered login backend.

Every plugin backend reaches the browser vault only through ``PluginLoginBackend``, so the isolation
rules hold however the plugin implements ``LoginBackend``:

- Its namespace is ``<registered name>:``, taken from the registry key, never from attributes the plugin
  can change later. Ownership is decided here, so a plugin's own ``owns()`` is never consulted and it
  is never handed a handle outside its namespace (built-in ``vault_``/``op:``/``bw:`` or another plugin's).
- It is handed the handle and nothing else: no page, CDP session, task id, other entries or master
  password. ``needs_unlock`` is always False, so Hermes' master-password prompt never runs for it.
- Its exceptions are replaced by a fixed-text ``PluginBackendError`` raised outside the ``except``, so a
  secret in a plugin's error text reaches no tool result, log line or exception chain.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Optional

from agent.vault_backends.base import LoginBackend
from agent.vault_store import VAULT_KINDS, VaultItemMeta

_FAILED = object()


def _well_formed(meta: Any) -> bool:
    """``VaultItemMeta`` does not type-check its fields; the vault tool reads them outside any plugin guard."""
    return (isinstance(meta, VaultItemMeta) and meta.kind in VAULT_KINDS
            and all(isinstance(getattr(meta, f), str) for f in ("id", "label", "created_at"))
            and all(getattr(meta, f) is None or isinstance(getattr(meta, f), str)
                    for f in ("origin", "identifier", "identifier_type"))
            and isinstance(meta.has_otp, bool) and isinstance(meta.allowed_origins, tuple)
            and all(isinstance(o, str) for o in meta.allowed_origins))


class PluginBackendError(RuntimeError):
    """A plugin login backend failed or broke its contract (fixed text: never the plugin's own message)."""


class PluginLoginBackend(LoginBackend):
    needs_unlock = False

    def __init__(self, name: str, plugin: LoginBackend):
        self.name, self.prefix, self._plugin = name, f"{name}:", plugin
        self.display_name = f"{name} (plugin)"  # core-composed: a plugin cannot present itself as a built-in manager

    def owns(self, handle: str) -> bool:
        return isinstance(handle, str) and handle.startswith(self.prefix)

    def _call(self, fn: Callable[[], Any]) -> Any:
        try:
            result = fn()
        except Exception:  # health: allow BLE001 -- the plugin's error text may hold a credential; replaced below
            result = _FAILED
        if result is _FAILED:
            raise PluginBackendError(f"The {self.name} login backend failed.")
        return result

    def _require_own(self, handle: str) -> None:
        if not self.owns(handle):
            raise PluginBackendError(f"Handle is outside the {self.name} login backend's namespace.")

    def _require(self, ok: bool) -> None:
        if not ok:
            raise PluginBackendError(f"The {self.name} login backend returned an invalid value.")

    def list_items(self) -> list[VaultItemMeta]:
        items = self._call(lambda: list(self._plugin.list_items()))
        self._require(all(_well_formed(m) for m in items))
        return [m for m in items if self.owns(m.id)]

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        if not self.owns(handle):
            return None
        meta = self._call(lambda: self._plugin.get_meta(handle))
        self._require(meta is None or _well_formed(meta))
        return meta if meta is not None and meta.id == handle else None

    def resolve_password(self, handle: str) -> str:
        self._require_own(handle)
        value = self._call(lambda: self._plugin.resolve_password(handle))
        self._require(isinstance(value, str))
        return value

    def resolve_otp(self, handle: str) -> Optional[str]:
        self._require_own(handle)
        value = self._call(lambda: self._plugin.resolve_otp(handle))
        self._require(value is None or isinstance(value, str))
        return value

    def resolve_secret(self, handle: str) -> dict[str, str]:
        self._require_own(handle)
        value = self._call(lambda: self._plugin.resolve_secret(handle))
        self._require(isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                       for k, v in value.items()))
        return dict(value)

    @contextmanager
    def fill_context(self):
        """The plugin's own context around one of ITS fills; it never sees the core's exception."""
        context = self._call(lambda: self._plugin.fill_context())
        self._call(lambda: context.__enter__())
        try:
            yield
        finally:
            self._call(lambda: context.__exit__(None, None, None))
