"""Profile-scoped browser-vault login backends registered by trusted in-process plugins.

Registration (``ctx.register_login_backend``) is refused for: a name that is not a lowercase
identifier, a handle prefix other than ``<name>:``, a name or prefix in a built-in backend's namespace,
a name another plugin already holds in this profile, a backend that wants Hermes' master-password
prompt (``needs_unlock``), and ``name``/``prefix`` computed by a descriptor (they could change after
validation). ``enabled_plugin_backends`` re-checks every stored key and wraps each backend in
:class:`agent.vault_backends.plugin.PluginLoginBackend`, the only form the vault tool ever sees.
"""

import inspect
import logging
import re

from agent.provider_registry import ProviderRegistry
from agent.vault_backends.base import LoginBackend

_registry = ProviderRegistry(label="Login backend", provider_cls=LoginBackend,
                             logger=logging.getLogger(__name__))
_registry.export(globals())

_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _builtin_namespaces() -> tuple[set[str], tuple[str, ...]]:
    from agent.vault_backends.base import external_backend_classes
    from agent.vault_backends.local import LocalLoginBackend

    classes = (LocalLoginBackend, *external_backend_classes())
    names = {cls.name for cls in classes} | {cls.prefix.rstrip(":_") for cls in classes}
    return names, tuple(cls.prefix for cls in classes)


def _namespace_error(name) -> str:
    """Why ``name`` cannot be a plugin backend's namespace, or "" when it can."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        return "login backend name must be a lowercase identifier"
    names, prefixes = _builtin_namespaces()
    if name in names or f"{name}:".startswith(prefixes):
        return "builtin login backend namespace is reserved"
    return ""


def register_provider(provider, *, scope=None):
    if not isinstance(provider, LoginBackend):
        raise ValueError("login backend must subclass LoginBackend")
    name, prefix = inspect.getattr_static(provider, "name", None), inspect.getattr_static(provider, "prefix", None)
    if not isinstance(name, str) or not isinstance(prefix, str):
        raise ValueError("login backend name and prefix must be plain string attributes")
    error = _namespace_error(name) or ("" if prefix == f"{name}:" else "login backend prefix must be '<name>:'")
    if error:
        raise ValueError(error)
    if inspect.getattr_static(provider, "needs_unlock", False) is not False:
        raise ValueError("plugin login backends cannot use the Hermes master-password prompt (needs_unlock)")
    if name in _registry.merged(scope):
        raise ValueError(f"login backend name {name!r} is already registered")
    _registry.register(provider, scope=scope)


def enabled_plugin_backends(cfg: dict) -> list[LoginBackend]:
    """Registered backends the profile turned on with ``vault.<name>.enabled: true``, wrapped."""
    from agent.vault_backends.plugin import PluginLoginBackend

    out: list[LoginBackend] = []
    for name, provider in sorted(_registry.merged().items()):
        section = cfg.get(name)
        if not _namespace_error(name) and isinstance(section, dict) and section.get("enabled") is True:
            out.append(PluginLoginBackend(name, provider))
    return out
