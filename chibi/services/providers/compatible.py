"""Factory for dynamically created, user-configured compatible provider instances.

Compatible providers (OpenAICompatible, AnthropicCompatible, ...) are created
at package import time from environment variables of the shape
``<PREFIX>_<NAME>_API_KEY`` / ``<PREFIX>_<NAME>_URL``. Each instance becomes a
concrete ``Provider`` subclass and is registered automatically by
``Provider.__init_subclass__`` when ``type()`` instantiates it.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from loguru import logger

from chibi.services.providers.provider import Provider, RegisteredProviders

INSTANCE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
ENV_INSTANCE_VAR_PATTERN = "{prefix}_(.+)_API_KEY"


def validate_instance_name(name: str, existing_keys: Iterable[str]) -> None:
    """Validate a compatible-instance name against the charset and registry keys.

    Args:
        name: Instance name, used verbatim as the provider display name.
        existing_keys: Lowercased registry keys that are already taken.

    Raises:
        ValueError: If the name is empty, contains characters outside
            ``[A-Za-z0-9_]`` or collides case-insensitively with an existing
            registry key.
    """
    if not INSTANCE_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid provider instance name {name!r}: only letters, digits and underscores are allowed")
    if name.lower() in {key.lower() for key in existing_keys}:
        raise ValueError(f"Provider instance name {name!r} collides with the existing registry key {name.lower()!r}")


def create_compatible_provider(
    *,
    base_class: type[Provider],
    name: str,
    api_key: str | None,
    base_url: str,
    class_name: str | None = None,
    dynamic: bool = True,
) -> type[Provider]:
    """Create and register a named compatible provider subclass.

    ``type()`` fires ``Provider.__init_subclass__``, so the returned class is
    registered in ``RegisteredProviders.all`` under the lowercased ``name`` and
    additionally in ``RegisteredProviders.available`` when ``api_key`` is
    truthy. The caller owns registry cleanup.

    Args:
        base_class: Base class to subclass. It must NOT define a ``name``
            class attribute (annotation-only), otherwise the base itself
            self-registers.
        name: Instance display name; the registry key is its lowercase form.
        api_key: API key snapshot; a falsy key leaves the instance registered
            but not available.
        base_url: API endpoint base URL.
        class_name: Optional concrete class name; derived from the instance
            name when omitted.
        dynamic: Marks instances created from the environment; static
            baseline instances (such as CustomOpenAI) pass False.

    Returns:
        The freshly created and registered provider class.

    Raises:
        ValueError: If the name fails validation (see ``validate_instance_name``).
    """
    validate_instance_name(name, RegisteredProviders.all)
    resolved_class_name = class_name or f"{base_class.__name__}_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
    attrs: dict[str, Any] = {
        "name": name,
        "api_key": api_key,
        "base_url": base_url,
        "is_dynamic_provider": dynamic,
    }
    return type(resolved_class_name, (base_class,), attrs)


def collect_compatible_configs(
    env: Mapping[str, str], *, prefix: str, default_base_url: str
) -> dict[str, dict[str, str]]:
    """Collect compatible-instance configs from a mapping.

    Args:
        env: Mapping to scan, usually ``os.environ``.
        prefix: Environment prefix, e.g. ``OPENAICOMPATIBLE``.
        default_base_url: Endpoint used when ``<PREFIX>_<NAME>_URL`` is absent
            or empty.

    Returns:
        Mapping of instance name to ``{"api_key": ..., "base_url": ...}``.
        Entries with an empty key value or an invalid name are skipped.
    """
    configs: dict[str, dict[str, str]] = {}
    pattern = re.compile(ENV_INSTANCE_VAR_PATTERN.format(prefix=prefix))
    for variable, value in env.items():
        match = pattern.fullmatch(variable)
        if not match:
            continue
        name = match.group(1)
        if not value:
            logger.debug(f"{prefix}: skipping {variable}: no API key configured")
            continue
        if not INSTANCE_NAME_PATTERN.fullmatch(name):
            logger.warning(f"{prefix}: skipping {variable}: invalid instance name {name!r}")
            continue
        url_variable = f"{prefix}_{name}_URL"
        configs[name] = {
            "api_key": value,
            "base_url": env.get(url_variable) or default_base_url,
        }
    return configs


def register_compatible_instances(
    *,
    env: Mapping[str, str],
    prefix: str,
    default_base_url: str,
    factory: Callable[..., type[Provider]],
) -> None:
    """Create and register compatible instances from environment variables.

    Invalid or colliding names are skipped with a warning instead of failing
    the startup. Instances are processed in a deterministic order (by registry
    key first, then by the raw name), so the surviving instance of a case-only
    name collision does not depend on the iteration order of ``env``.

    Args:
        env: Mapping to scan, usually ``os.environ``.
        prefix: Environment prefix, e.g. ``OPENAICOMPATIBLE``.
        default_base_url: Endpoint used when the per-instance URL is absent.
        factory: Zero-config callable accepting ``name``, ``api_key`` and
            ``base_url`` keyword arguments.
    """
    configs = collect_compatible_configs(env=env, prefix=prefix, default_base_url=default_base_url)
    seen: set[str] = set()
    for name in sorted(configs, key=lambda candidate: (candidate.lower(), candidate)):
        key = name.lower()
        if key in seen:
            logger.warning(f"{prefix}: skipping instance {name!r}: case-insensitive name collision")
            continue
        seen.add(key)
        try:
            factory(name=name, api_key=configs[name]["api_key"], base_url=configs[name]["base_url"])
        except ValueError as error:
            logger.warning(f"{prefix}: skipping instance {name!r}: {error}")
