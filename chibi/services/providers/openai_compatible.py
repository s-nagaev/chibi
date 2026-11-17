"""OpenAI-compatible providers defined through environment variables."""

import os
from collections.abc import Mapping

from chibi.services.providers.compatible import (
    collect_compatible_configs,
    create_compatible_provider,
    register_compatible_instances,
)
from chibi.services.providers.provider import OpenAIFriendlyProvider, Provider

OPENAI_COMPATIBLE_ENV_PREFIX = "OPENAICOMPATIBLE"
DEFAULT_OPENAI_COMPATIBLE_URL = "http://localhost:1234/v1"


class OpenAICompatible(OpenAIFriendlyProvider):
    """Base class for user-configured OpenAI-compatible endpoints.

    Concrete instances are created dynamically by the compatible-provider
    factory. This base must never define a ``name`` class attribute, otherwise
    it would self-register as a provider.
    """

    chat_ready = True
    tts_ready = True
    stt_ready = True
    default_model = ""


def create_openai_compatible(
    *,
    name: str,
    api_key: str | None,
    base_url: str,
    class_name: str | None = None,
    dynamic: bool = True,
) -> type[Provider]:
    """Create and register a named OpenAI-compatible instance.

    Args:
        name: Instance display name; the registry key is its lowercase form.
        api_key: API key snapshot; a falsy key leaves the instance registered
            but not available.
        base_url: OpenAI-compatible endpoint base URL.
        class_name: Optional concrete class name; derived from the instance
            name when omitted.
        dynamic: Marks instances created from the environment; the static
            CustomOpenAI baseline instance passes False.

    Returns:
        The freshly created and registered provider class.

    Raises:
        ValueError: If the name fails validation (see ``validate_instance_name``).
    """
    return create_compatible_provider(
        base_class=OpenAICompatible,
        name=name,
        api_key=api_key,
        base_url=base_url,
        class_name=class_name,
        dynamic=dynamic,
    )


def collect_openai_compatible_configs(env: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Collect OpenAI-compatible instance configs from a mapping.

    Args:
        env: Mapping to scan, usually ``os.environ``.

    Returns:
        Mapping of instance name to ``{"api_key": ..., "base_url": ...}``.
    """
    return collect_compatible_configs(
        env=env,
        prefix=OPENAI_COMPATIBLE_ENV_PREFIX,
        default_base_url=DEFAULT_OPENAI_COMPATIBLE_URL,
    )


def register_openai_compatible_instances(env: Mapping[str, str] | None = None) -> None:
    """Register OpenAI-compatible instances from the environment.

    Args:
        env: Mapping to scan; defaults to ``os.environ``.
    """
    register_compatible_instances(
        env=os.environ if env is None else env,
        prefix=OPENAI_COMPATIBLE_ENV_PREFIX,
        default_base_url=DEFAULT_OPENAI_COMPATIBLE_URL,
        factory=create_openai_compatible,
    )
