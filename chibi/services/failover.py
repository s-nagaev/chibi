"""Core provider failover engine.

Implements the owner-confirmed failover semantics (see
``.project/provider-fallback/plan.md`` in the main repository):

AUTO mode (default, zero config):
    1. When provider X fails to serve model Y, provider X is put on a
       20-minute cooldown. The cooldown affects ONLY auto-selection (the
       failover ladder); it never blocks manual choices (explicit user
       model selection or manual chains).
    2. If model Y exists at another provider (exact name match, source of
       truth = in-memory model registry, never a live per-request
       enumeration) -> retry there.
    3. If Y exists nowhere else -> retry on the next provider's DEFAULT
       model (``model=None`` keeps the existing default-model semantics).
    4. If nothing works -> fail with an honest error. No model-mixing, no
       cross-mode cascades.

MANUAL mode (opt-in, per role):
    An ordered chain of (provider, model) pairs. Head of the chain is the
    primary. The chain fully replaces the auto ladder for that role. The
    cooldown never blocks a manual chain (explicit user choice).

DISABLED mode:
    No fallback at all for the role: a single attempt, honest failure.

Hard rules honoured here:

- No ``context_overflow`` trigger (two-tier protection already exists) and
  no moderation fallback: those exceptions are deliberately NOT classified
  as failover triggers.
- No revisiting already-tried (provider, model) pairs within one request.
- Cooldown is the ONLY persistent state (Redis-backed, 20 min fixed TTL,
  not configurable in v1).
- Happy path stays clean: the primary is attempted first, and the auto
  ladder — including provider model enumeration and any cooldown-store
  access — is resolved lazily, only when the primary fails.
- Enumeration-failure interpretation: a provider whose model list could
  not be enumerated is skipped as an exact-match (step 2) candidate but is
  still a step 3 (default-model) candidate — an unknown model list is not
  an absent model.
- Latency note: in-provider retries (tenacity, ``reraise=True``) can burn
  ~5-10 minutes per provider before failover fires; the engine simply sees
  the re-raised raw/service exception after the provider gave up.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Optional

import aiohttp
import httpx
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import InternalServerError as AnthropicInternalServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ServerError as GeminiServerError
from loguru import logger
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import InternalServerError as OpenAIInternalServerError
from openai import RateLimitError as OpenAIRateLimitError

from chibi.exceptions import (
    ContextLengthExceededError,
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NoProviderSelectedError,
    NoResponseError,
    NotAuthorizedError,
    ServiceConnectionError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.utils.app import SingletonMeta

if TYPE_CHECKING:
    from chibi.models import Message, User
    from chibi.schemas.app import ChatResponseSchema
    from chibi.services.providers.provider import Provider

Role = Literal["master", "subagent"]
TriggerKind = Literal["rate_limit", "server_error", "timeout", "network"]

COOLDOWN_TTL_SECONDS = 20 * 60  # 20 minutes, fixed per owner decision (v1)
COOLDOWN_KEY_PREFIX = "failover:cooldown:"

ChatCall = Callable[["Provider", Optional[str]], Awaitable[tuple["ChatResponseSchema", list["Message"]]]]

# Failover trigger taxonomy. Mapping is checked top-down; order matters
# because SDK exception hierarchies overlap (e.g. openai.APITimeoutError is
# a subclass of APIConnectionError, httpx.TimeoutException is a subclass of
# TransportError). Context-length and auth/config errors are intentionally
# NOT failover triggers.
_TIMEOUT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    OpenAIAPITimeoutError,
    AnthropicAPITimeoutError,
    aiohttp.ServerTimeoutError,
    TimeoutError,
)
_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    OpenAIAPIConnectionError,
    AnthropicAPIConnectionError,
    aiohttp.ClientError,
    ConnectionError,
)
_RATE_LIMIT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ServiceRateLimitError,
    OpenAIRateLimitError,
    AnthropicRateLimitError,
)
_SERVER_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ServiceResponseError,
    NoResponseError,
    ServiceConnectionError,
    OpenAIInternalServerError,
    AnthropicInternalServerError,
    GeminiServerError,
)
_NOT_FAILOVER_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ContextLengthExceededError,  # context_overflow trigger is explicitly out of scope
    NotAuthorizedError,
    NoApiKeyProvidedError,
    NoProviderSelectedError,
    NoModelSelectedError,
)


class FailoverTrigger(Exception):
    """A provider failure classified as failover-worthy."""

    def __init__(self, kind: TriggerKind, provider: str, model: str | None, original: BaseException) -> None:
        self.kind: TriggerKind = kind
        self.provider = provider
        self.model = model
        self.original = original
        super().__init__(f"{kind} from {provider}/{model or '<default>'}: {original}")

    def __str__(self) -> str:
        return f"FailoverTrigger(kind={self.kind!r}, provider={self.provider!r}, model={self.model!r})"


def classify_exception(exception: BaseException) -> TriggerKind | None:
    """Map a raw provider/service exception to a failover trigger kind.

    Tenacity retry policies use ``reraise=True`` on every chat path, so the
    engine sees the final provider exception, not ``RetryError``. Both raw
    SDK exceptions and Chibi service exceptions (produced by the provider
    error-conversion wrappers) are recognised.

    Args:
        exception: The exception raised by the provider call.

    Returns:
        One of ``rate_limit`` / ``server_error`` / ``timeout`` / ``network``,
        or None when the exception must NOT trigger failover (context
        overflow, auth/config problems, or unrelated errors).
    """
    if isinstance(exception, _NOT_FAILOVER_EXCEPTIONS):
        return None
    if isinstance(exception, _RATE_LIMIT_EXCEPTIONS):
        return "rate_limit"
    if isinstance(exception, _TIMEOUT_EXCEPTIONS):
        return "timeout"
    if isinstance(exception, _NETWORK_EXCEPTIONS):
        return "network"
    if isinstance(exception, _SERVER_EXCEPTIONS):
        return "server_error"
    return None


class FailoverMode(str, Enum):
    AUTO = "auto"
    DISABLED = "disabled"
    MANUAL = "manual"


@dataclass(frozen=True)
class FailoverPair:
    """One (provider, model) step of a failover chain.

    ``model=None`` means "provider's default model" (existing semantics).
    """

    provider: str
    model: str | None = None


@dataclass
class FailoverPolicy:
    """Resolved failover behaviour for one role (master / subagent)."""

    mode: FailoverMode = FailoverMode.AUTO
    chain: list[FailoverPair] = field(default_factory=list)


def resolve_failover_policy(role: Role) -> FailoverPolicy:
    """Resolve the failover policy for a role.

    v1 bootstrap default: AUTO for every role. The ``failover_config`` task
    replaces this with the real resolution order (``FAILOVER_CHAIN_MASTER``
    / ``FAILOVER_CHAIN_SUBAGENT`` env bootstrap defaults + Redis runtime
    override), keeping this seam intact.

    Args:
        role: ``"master"`` or ``"subagent"``.

    Returns:
        The policy for the role.
    """
    return FailoverPolicy(mode=FailoverMode.AUTO, chain=[])


class CooldownStore(metaclass=SingletonMeta):
    """Persistent 20-minute provider cooldown (the ONLY failover state).

    Redis-backed through the existing storage layer (the ``RedisStorage``
    driver is used as-is, never modified): key
    ``failover:cooldown:<provider>`` with a fixed 1200s TTL. When the
    storage backend is not Redis (or Redis is momentarily unreachable) the
    store degrades to a per-process in-memory map so failover keeps
    working; cooldown sharing across replicas is a Redis-only feature.

    The cooldown affects ONLY auto-selection (the auto ladder). Manual
    chains and explicit user selections are never blocked by it.

    Storage resolution is cached on the (process-wide) singleton: the
    app-wide ``chibi.storage.database._db_provider`` ``DatabaseCache``
    instance is reused (never a fresh ``DatabaseCache()`` per operation),
    so exactly ONE storage resolution happens per process and the resolved
    Redis client is reused for every subsequent cooldown operation.
    """

    def __init__(self) -> None:
        self._memory: dict[str, float] = {}
        self._client: Any | None = None
        self._client_resolved = False
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(provider_name: str) -> str:
        return f"{COOLDOWN_KEY_PREFIX}{provider_name.lower()}"

    async def _redis_client(self) -> Any | None:
        """Return the cached Redis client, or None when unavailable.

        The app-wide database instance (``_db_provider``) is resolved at
        most once per process; a transient resolution failure is not cached
        (the store fail-opens to memory and retries on a later operation).
        """
        if self._client_resolved:
            return self._client
        async with self._lock:
            if self._client_resolved:
                return self._client
            try:
                from chibi.storage.database import _db_provider  # app-wide instance
                from chibi.storage.redis import RedisStorage

                db = await _db_provider.get_database()
            except Exception as e:  # storage not configured / unavailable
                logger.debug(f"Cooldown store: no database available ({e})")
                return None
            self._client = db.redis if isinstance(db, RedisStorage) else None
            self._client_resolved = True
            return self._client

    async def mark(self, provider_name: str) -> None:
        """Put the provider on a fixed 20-minute cooldown."""
        key = self._key(provider_name)
        client = await self._redis_client()
        if client is not None:
            try:
                await client.set(name=key, value=str(time.time()), ex=COOLDOWN_TTL_SECONDS)
                return
            except Exception as e:
                logger.warning(f"Cooldown store: Redis write failed, falling back to memory ({e})")
        self._memory[key] = time.monotonic() + COOLDOWN_TTL_SECONDS

    async def is_blocked(self, provider_name: str) -> bool:
        """Return True when the provider is on cooldown (auto-selection only)."""
        key = self._key(provider_name)
        client = await self._redis_client()
        if client is not None:
            try:
                return bool(await client.get(key))
            except Exception as e:
                logger.warning(f"Cooldown store: Redis read failed, falling back to memory ({e})")
        expires_at = self._memory.get(key)
        if expires_at is None:
            return False
        if expires_at <= time.monotonic():
            self._memory.pop(key, None)
            return False
        return True


class FailoverModelRegistry(metaclass=SingletonMeta):
    """In-memory provider -> model-names map for auto-ladder step 2.

    The source of truth for "does model Y exist at provider P" is this
    in-memory map, NOT a live per-request enumeration (provider model-list
    APIs can be slow or fail). The map is filled lazily from
    ``provider.get_available_models()`` and cached with a TTL. When the
    enumeration fails for a provider, that provider is skipped (marker
    cached for a shorter TTL) and the ladder continues.
    """

    _TTL_SECONDS = 60 * 60
    _FAILURE_TTL_SECONDS = 5 * 60

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, set[str] | None]] = {}

    async def models_for(self, provider: "Provider") -> set[str] | None:
        """Return the chat-ready model names of the provider.

        Args:
            provider: Provider instance.

        Returns:
            A set of exact model names, or None when the enumeration is
            unavailable/failed (caller must skip this provider).
        """
        key = provider.name.lower()
        now = time.monotonic()
        entry = self._cache.get(key)
        if entry and entry[0] > now:
            return entry[1]
        try:
            models = await provider.get_available_models(image_generation=False)
        except Exception as e:
            logger.warning(f"Failover model registry: failed to enumerate models of {provider.name}: {e}")
            self._cache[key] = (now + self._FAILURE_TTL_SECONDS, None)
            return None
        names = {model.name for model in models}
        self._cache[key] = (now + self._TTL_SECONDS, names)
        return names


@dataclass
class FailoverAttempt:
    """A single (provider, model) attempt outcome within one request."""

    provider: str
    model: str | None
    ok: bool
    kind: TriggerKind | None = None
    error: str | None = None


@dataclass
class FailoverResult:
    """Outcome of a failover-guarded chat request.

    ``fallback_used`` / ``serving_provider`` / ``serving_model`` are what
    the notification layer (failover_notification task) consumes to warn
    the user honestly about any fallback.
    """

    response: "ChatResponseSchema"
    new_messages: list["Message"]
    fallback_used: bool = False
    serving_provider: str = ""
    serving_model: str | None = None
    attempts: list[FailoverAttempt] = field(default_factory=list)
    last_trigger: FailoverTrigger | None = None


class FailoverEngine:
    """Runs a chat call behind the failover semantics for one role."""

    def __init__(self, user: "User") -> None:
        self.user = user
        self.cooldown = CooldownStore()
        self.model_registry = FailoverModelRegistry()

    # -- candidate resolution ------------------------------------------------

    def _chat_ready_instances(self) -> dict[str, "Provider"]:
        """Instantiate every chat-ready provider available to this user.

        Degrades to an empty map (primary-only ladder) when the user object
        carries no usable provider registry (e.g. minimal test doubles).
        """
        instances: dict[str, "Provider"] = {}
        try:
            registered = self.user.providers
            provider_classes = registered.chat_ready
            for name, provider_class in provider_classes.items():
                instance = registered.get_instance(provider_class)
                if instance is not None:
                    instances[name.lower()] = instance
        except (AttributeError, TypeError):
            return {}
        return instances

    async def _build_auto_ladder(
        self,
        primary_provider: "Provider",
        model: str | None,
        instances: dict[str, "Provider"],
        blocked: dict[str, bool],
    ) -> list[FailoverPair]:
        """Build the ordered auto ladder for (primary, model).

        Step 2 (exact model match elsewhere) precedes step 3 (next
        provider's default model); step 3 only applies when model Y exists
        NOWHERE else. Providers on cooldown are skipped; the primary is
        always attempted first because it is the user's active selection
        (a cooldown never blocks an explicit choice, and a re-failure
        simply re-marks the cooldown).

        Documented interpretation of the plan's enumeration-failure rule:
        it governs STEP 2 only — a provider whose model list could not be
        enumerated ("model list unknown") is skipped as an exact-match
        candidate but is NOT excluded from step 3, where it is attempted
        with its default model (``model=None``), because an unknown model
        list does not mean the model is absent.

        Args:
            primary_provider: The active provider resolved by the caller.
            model: The active model (None = provider default).
            instances: Chat-ready provider instances keyed by lowercase name
                (the primary is guaranteed to be present).
            blocked: Cooldown map (populated in place, lazily).

        Returns:
            The ordered ladder of (provider, model) pairs to attempt.
        """
        primary_name = primary_provider.name if isinstance(primary_provider.name, str) else "primary"
        ladder = [FailoverPair(provider=primary_name, model=model)]

        candidates: list[tuple[str, "Provider"]] = []
        for name, instance in instances.items():
            if name == primary_name.lower():
                continue
            if name not in blocked:
                blocked[name] = await self.cooldown.is_blocked(name)
            if blocked[name]:
                logger.debug(f"Failover: skipping {name} (on cooldown)")
                continue
            candidates.append((name, instance))

        # Step 2: providers that have the exact same model (in-memory
        # registry map; enumeration failure -> skip that provider).
        with_model: list[str] = []
        if model:
            for name, instance in candidates:
                names = await self.model_registry.models_for(instance)
                if names is None:
                    continue
                if model in names:
                    with_model.append(name)
        for name in with_model:
            ladder.append(FailoverPair(provider=instances[name].name, model=model))

        # Step 3: default models, ONLY when the model exists nowhere else.
        if model and not with_model:
            for name, instance in candidates:
                ladder.append(FailoverPair(provider=instance.name, model=None))

        return ladder

    # -- execution -----------------------------------------------------------

    async def run(
        self,
        role: Role,
        primary_provider: "Provider",
        model: str | None,
        call: ChatCall,
    ) -> FailoverResult:
        """Execute ``call`` behind the failover policy for ``role``.

        Args:
            role: ``"master"`` or ``"subagent"``.
            primary_provider: The active provider resolved by the caller.
            model: The active model (None = provider default).
            call: Callable receiving (provider, model) and returning the
                provider-call coroutine with the caller's fixed arguments.

        Returns:
            A ``FailoverResult`` (response + fallback metadata).

        Raises:
            Exception: The last provider exception when every step of the
                chain fails (honest failure) — or immediately when the
                exception is not a failover trigger.
        """
        policy = resolve_failover_policy(role)
        tried: set[tuple[str, str | None]] = set()
        result = FailoverResult(response=None, new_messages=[])  # type: ignore[arg-type]
        last_exception: BaseException | None = None
        last_trigger: FailoverTrigger | None = None
        # Guard against non-string provider names (e.g. AsyncMock doubles in
        # tests): every real provider defines ``name`` as a class-level str.
        primary_name = primary_provider.name if isinstance(primary_provider.name, str) else "primary"
        primary_key = primary_name.lower()
        first_attempted: FailoverPair | None = None

        # Chat-ready provider instances for this user. Built LAZILY: a
        # fully successful request never instantiates extra providers nor
        # touches the cooldown store — the ladder (and the model
        # enumeration behind it) is resolved only when the primary fails.
        instances: dict[str, "Provider"] | None = None

        async def get_instances() -> dict[str, "Provider"]:
            nonlocal instances
            if instances is None:
                instances = self._chat_ready_instances()
                # The primary is always included so the first attempt never
                # depends on registry filtering.
                instances.setdefault(primary_key, primary_provider)
            return instances

        async def attempt(pair: FailoverPair, provider_instance: "Provider") -> bool:
            """Run one (provider, model) pair. True on success."""
            nonlocal last_exception, last_trigger, first_attempted
            key = (pair.provider.lower(), pair.model)
            if key in tried:
                return False
            tried.add(key)
            if first_attempted is None:
                first_attempted = pair
            try:
                response, new_messages = await call(provider_instance, pair.model)
            except Exception as e:
                kind = classify_exception(e)
                result.attempts.append(
                    FailoverAttempt(provider=pair.provider, model=pair.model, ok=False, kind=kind, error=str(e))
                )
                if kind is None:
                    raise  # not failover-worthy (context overflow, auth, ...) — propagate honestly
                last_exception = e
                last_trigger = FailoverTrigger(kind=kind, provider=pair.provider, model=pair.model, original=e)
                await self.cooldown.mark(pair.provider)
                logger.warning(
                    f"Failover[{role}]: {pair.provider}/{pair.model or '<default>'} failed ({kind}); "
                    f"provider put on a {COOLDOWN_TTL_SECONDS // 60}-minute cooldown."
                )
                return False
            result.response = response
            result.new_messages = new_messages
            result.serving_provider = response.provider or pair.provider
            result.serving_model = pair.model or response.model
            # Any successful pair other than the first attempted one (the
            # primary) is by definition a fallback.
            result.fallback_used = pair != first_attempted
            result.attempts.append(FailoverAttempt(provider=pair.provider, model=pair.model, ok=True))
            return True

        if policy.mode == FailoverMode.DISABLED:
            await attempt(FailoverPair(provider=primary_name, model=model), primary_provider)
            if result.response is None:
                raise last_exception or last_trigger or RuntimeError("Failover: attempt produced no result")
            return result

        if policy.mode == FailoverMode.MANUAL:
            # Head of chain = primary; the chain fully replaces auto and is
            # never blocked by the cooldown. An empty chain falls back to a
            # single attempt on the caller-resolved primary.
            pairs_iter: list[FailoverPair] = list(policy.chain) or [FailoverPair(provider=primary_name, model=model)]
            cooldown_check = False
            instances_map = await get_instances()
        else:  # AUTO
            # Primary attempt FIRST: the auto ladder — including provider
            # model enumeration — is resolved lazily, only when the primary
            # actually fails. A successful request pays for neither the
            # cooldown store nor the model registry.
            if await attempt(FailoverPair(provider=primary_name, model=model), primary_provider):
                return result
            blocked: dict[str, bool] = {}
            instances_map = await get_instances()
            pairs_iter = await self._build_auto_ladder(
                primary_provider=primary_provider, model=model, instances=instances_map, blocked=blocked
            )
            cooldown_check = True

        for index, pair in enumerate(pairs_iter):
            instance = instances_map.get(pair.provider.lower())
            if instance is None:
                logger.warning(
                    f"Failover[{role}]: chain step {pair.provider}/{pair.model or '<default>'} "
                    f"is not available to this user — skipping."
                )
                continue
            # No revisiting already-tried pairs within one request.
            if (pair.provider.lower(), pair.model) in tried:
                continue
            # Cooldown blocks auto-selection only, never the primary and
            # never manual chains. Cooldowns set during THIS request are
            # honoured for later ladder steps as well.
            if (
                cooldown_check
                and index > 0
                and pair.provider.lower() != primary_key
                and await self.cooldown.is_blocked(pair.provider)
            ):
                logger.debug(f"Failover[{role}]: skipping {pair.provider} (on cooldown)")
                continue
            if await attempt(pair, instance):
                return result

        # Nothing worked — fail honestly with the original exception.
        if result.response is not None:  # pragma: no cover — defensive
            return result
        raise last_exception or last_trigger or RuntimeError("Failover: no provider could serve the request")


async def run_with_failover(
    role: Role,
    user: "User",
    primary_provider: "Provider",
    model: str | None,
    call: ChatCall,
) -> FailoverResult:
    """Convenience entry point used by the integration call sites.

    Args:
        role: ``"master"`` (user.py answer path) or ``"subagent"``
            (``get_sub_agent_response``).
        user: The requesting user (used for provider registry access).
        primary_provider: Active provider resolved by the caller.
        model: Active model, or None for the provider default.
        call: Callable ``(provider, model) -> coroutine`` performing the
            actual ``provider.get_chat_response(...)`` with the caller's
            fixed keyword arguments.

    Returns:
        The ``FailoverResult`` with the response and fallback metadata.
    """
    return await FailoverEngine(user=user).run(role=role, primary_provider=primary_provider, model=model, call=call)
