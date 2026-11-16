"""Anthropic-compatible providers defined through environment variables."""

import os
from collections.abc import Mapping

from anthropic import AsyncClient

from chibi.exceptions import NoApiKeyProvidedError
from chibi.services.providers.compatible import (
    collect_compatible_configs,
    create_compatible_provider,
    register_compatible_instances,
)
from chibi.services.providers.provider import AnthropicFriendlyProvider, Provider

ANTHROPIC_COMPATIBLE_ENV_PREFIX = "ANTHROPICCOMPATIBLE"
DEFAULT_ANTHROPIC_COMPATIBLE_URL = "https://api.anthropic.com"


class AnthropicCompatible(AnthropicFriendlyProvider):
    """Base class for user-configured Anthropic-compatible endpoints.

    Concrete instances are created dynamically by the compatible-provider
    factory. This base must never define a ``name`` class attribute, otherwise
    it would self-register as a provider.

    Requests follow the native Anthropic contract: ``x-api-key`` and
    ``anthropic-version`` headers (handled by the Anthropic SDK client) against
    ``base_url``, which every instance overrides with its configured endpoint.
    """

    chat_ready = True
    default_model = ""

    def __init__(self, token: str) -> None:
        self._client: AsyncClient | None = None
        super().__init__(token=token)

    @property
    def client(self) -> AsyncClient:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = AsyncClient(api_key=self.token, base_url=self.base_url)
        return self._client

    @client.setter
    def client(self, value: AsyncClient) -> None:
        """Setter for client property to allow mocking in tests."""
        self._client = value

    def get_client(self) -> AsyncClient:
        """Get the client, checking for mock first."""
        return self.client


def create_anthropic_compatible(
    *,
    name: str,
    api_key: str | None,
    base_url: str,
    class_name: str | None = None,
    dynamic: bool = True,
) -> type[Provider]:
    """Create and register a named Anthropic-compatible instance.

    Args:
        name: Instance display name; the registry key is its lowercase form.
        api_key: API key snapshot; a falsy key leaves the instance registered
            but not available.
        base_url: Anthropic-compatible endpoint base URL.
        class_name: Optional concrete class name; derived from the instance
            name when omitted.
        dynamic: Marks instances created from the environment.

    Returns:
        The freshly created and registered provider class.

    Raises:
        ValueError: If the name fails validation (see ``validate_instance_name``).
    """
    return create_compatible_provider(
        base_class=AnthropicCompatible,
        name=name,
        api_key=api_key,
        base_url=base_url,
        class_name=class_name,
        dynamic=dynamic,
    )


def collect_anthropic_compatible_configs(env: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Collect Anthropic-compatible instance configs from a mapping.

    Args:
        env: Mapping to scan, usually ``os.environ``.

    Returns:
        Mapping of instance name to ``{"api_key": ..., "base_url": ...}``.
    """
    return collect_compatible_configs(
        env=env,
        prefix=ANTHROPIC_COMPATIBLE_ENV_PREFIX,
        default_base_url=DEFAULT_ANTHROPIC_COMPATIBLE_URL,
    )


def register_anthropic_compatible_instances(env: Mapping[str, str] | None = None) -> None:
    """Register Anthropic-compatible instances from the environment.

    Args:
        env: Mapping to scan; defaults to ``os.environ``.
    """
    register_compatible_instances(
        env=os.environ if env is None else env,
        prefix=ANTHROPIC_COMPATIBLE_ENV_PREFIX,
        default_base_url=DEFAULT_ANTHROPIC_COMPATIBLE_URL,
        factory=create_anthropic_compatible,
    )
