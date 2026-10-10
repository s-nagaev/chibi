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
    Degenerate case: with a ``None`` primary model (user never selected a
    model) the AUTO ladder collapses to a single attempt — the
    ``model and not with_model`` guard in the ladder builder suppresses
    step 3. This is deliberate: without a selected model there is nothing
    to "fail over from" on the same-provider step.

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
import inspect
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

from chibi.config.gpt import gpt_settings
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

# Notification hooks (failover_notification task): invoked by the engine when
# a fallback step is ABOUT to be attempted and when every step is exhausted.
# Hooks may be sync or async callables; the engine awaits async ones but never
# lets a hook failure break the failover walk itself.
FallbackHook = Callable[["FailoverTrigger", "FailoverPair"], Any]
FailureHook = Callable[["FailoverTrigger", list["FailoverPair"]], Any]

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


def format_failover_pair(provider: str, model: str | None) -> str:
    """Render a (provider, model) pair for user-facing messages.

    Args:
        provider: Provider name.
        model: Model name, or None for the provider's default model.

    Returns:
        ``provider/model`` or ``provider/<default>`` when the model is None.
    """
    return f"{provider}/{model if model is not None else '<default>'}"


def failover_warning_message(trigger: "FailoverTrigger", fallback: "FailoverPair") -> str:
    """Build the in-chat warning for one fallback transition.

    Owner-confirmed shape: which trigger fired and which fallback target is
    being attempted, e.g. ``⚠️ openai/gpt-5.6-luna hit rate_limit — falling
    back to gemini/gemini-2.5-flash``.

    Args:
        trigger: The failover trigger that fired on the previous attempt.
        fallback: The (provider, model) pair about to be attempted.

    Returns:
        The warning text.
    """
    return (
        f"⚠️ {format_failover_pair(trigger.provider, trigger.model)} hit {trigger.kind} — "
        f"falling back to {format_failover_pair(fallback.provider, fallback.model)}"
    )


def failover_failure_message(trigger: "FailoverTrigger", fallback_targets: list["FailoverPair"]) -> str:
    """Build the in-chat message for the final honest failure.

    Sent when the failover walk actually attempted fallbacks and none of
    them succeeded (a single-attempt failure with no fallback candidates
    never produces this message — the standard error handling covers it).

    Args:
        trigger: The last failover trigger observed.
        fallback_targets: The fallback (provider, model) pairs that were
            actually attempted (not merely configured).

    Returns:
        The honest failure text.
    """
    if fallback_targets:
        targets = ", ".join(format_failover_pair(pair.provider, pair.model) for pair in fallback_targets)
        tail = f"all fallback attempts failed ({targets}). No fallback succeeded."
    else:
        tail = "no fallback succeeded."
    return f"⚠️ {format_failover_pair(trigger.provider, trigger.model)} hit {trigger.kind} — {tail}"


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


def parse_failover_chain(raw: str | None, role: Role) -> FailoverPolicy:
    """Parse one ``FAILOVER_CHAIN_*`` setting into a :class:`FailoverPolicy`.

    Grammar (owner-confirmed, see the plan): the value is exactly one of

    - ``auto`` (case-insensitive) — the auto ladder;
    - ``disabled`` (case-insensitive) — no fallback for the role;
    - a comma-separated ordered list of ``provider/model`` pairs, where the
      pair splits on the FIRST slash only (provider names contain no
      slashes; model names may, e.g. ``google/gemini-3.5-flash-lite``).

    Whitespace is tolerated everywhere. Validation is non-fatal by design
    (install-time rule): every invalid entry (empty, missing slash, empty
    provider or model part, duplicate pair) produces a clear warning and is
    skipped — the application never crashes on a malformed chain. An
    unset/blank value means the default policy: AUTO.

    Args:
        raw: The raw setting value (``None`` when unset).
        role: ``"master"`` or ``"subagent"`` — used for warnings only.

    Returns:
        The resolved policy (AUTO/DISABLED with an empty chain, or MANUAL
        with the validated chain; the chain holds the FALLBACK steps —
        the engine prepends the call site's primary as the head).
    """
    if raw is None or not raw.strip():
        return FailoverPolicy(mode=FailoverMode.AUTO, chain=[])

    value = raw.strip()
    lowered = value.lower()
    env_name = f"FAILOVER_CHAIN_{role.upper()}"

    if lowered == "auto":
        return FailoverPolicy(mode=FailoverMode.AUTO, chain=[])
    if lowered == "disabled":
        return FailoverPolicy(mode=FailoverMode.DISABLED, chain=[])

    chain: list[FailoverPair] = []
    seen: set[tuple[str, str | None]] = set()
    for index, entry in enumerate(value.split(","), start=1):
        entry = entry.strip()
        if not entry:
            logger.warning(f"Failover[{role}]: {env_name} entry #{index} is empty — skipped.")
            continue
        provider, _, model = entry.partition("/")  # first slash only
        provider = provider.strip()
        model = model.strip()
        if not provider or not model:
            logger.warning(
                f"Failover[{role}]: {env_name} entry #{index} '{entry}' is invalid — "
                f"expected 'provider/model' (split on the first slash) — skipped."
            )
            continue
        key = (provider.lower(), model)
        if key in seen:
            logger.warning(f"Failover[{role}]: {env_name} entry #{index} '{entry}' is a duplicate pair — skipped.")
            continue
        seen.add(key)
        chain.append(FailoverPair(provider=provider, model=model))

    return FailoverPolicy(mode=FailoverMode.MANUAL, chain=chain)


# Env settings are bootstrap defaults (changes go through the Redis
# runtime override, not the process environment): the resolved policy is
# memoized per role for the process lifetime. Tests and future runtime
# overrides can invalidate it with :func:`reset_failover_policy_cache`.
_policy_cache: dict[str, FailoverPolicy] = {}


def resolve_failover_policy(role: Role) -> FailoverPolicy:
    """Resolve the failover policy for a role.

    Reads the ``FAILOVER_CHAIN_MASTER`` / ``FAILOVER_CHAIN_SUBAGENT``
    bootstrap default (via ``GPTSettings``) and parses it with
    :func:`parse_failover_chain`. The result is memoized per role: the
    environment is a bootstrap default, so re-reading it per request would
    be misleading once a Redis runtime override lands.

    Args:
        role: ``"master"`` or ``"subagent"``.

    Returns:
        The policy for the role (AUTO by default).
    """
    cached = _policy_cache.get(role)
    if cached is not None:
        return cached
    raw = getattr(gpt_settings, f"failover_chain_{role}_raw", None)
    policy = parse_failover_chain(raw, role)
    _policy_cache[role] = policy
    logger.info(
        f"Failover[{role}]: failover policy resolved — mode={policy.mode.value}, "
        f"configured chain steps={len(policy.chain)}."
    )
    return policy


def reset_failover_policy_cache() -> None:
    """Drop the memoized policies (tests / future runtime overrides)."""
    _policy_cache.clear()


def validate_failover_chains() -> None:
    """Install-time validation of both chain settings.

    Parses both roles once, so malformed ``FAILOVER_CHAIN_*`` values
    produce their warnings at startup (config load) instead of on the
    first request. Never raises: invalid entries are warned about and
    skipped by :func:`parse_failover_chain`.
    """
    for role in ("master", "subagent"):
        resolve_failover_policy(role)


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

    ``fallback_used`` / ``serving_provider`` / ``serving_model`` describe
    the outcome for callers; user-facing fallback warnings are delivered
    through the engine's ``on_fallback`` / ``on_failure`` notification
    hooks (failover_notification task), not from this dataclass.
    """

    response: "ChatResponseSchema"
    new_messages: list["Message"]
    fallback_used: bool = False
    serving_provider: str = ""
    serving_model: str | None = None
    attempts: list[FailoverAttempt] = field(default_factory=list)
    last_trigger: FailoverTrigger | None = None


@dataclass
class _WalkState:
    """Mutable per-request state shared by the failover walk helpers.

    Extracted from ``FailoverEngine.run`` so the walk steps can live in
    private methods instead of closures mutating ``nonlocal`` variables.
    One instance per ``run()`` call; never shared across requests.
    """

    result: FailoverResult
    tried: set[tuple[str, str | None]] = field(default_factory=set)
    last_exception: BaseException | None = None
    last_trigger: FailoverTrigger | None = None
    first_attempted: FailoverPair | None = None
    instances: dict[str, "Provider"] | None = None


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

    @staticmethod
    async def _notify(hook_name: str, hook: Callable[..., Any], *args: Any) -> None:
        """Invoke a notification hook without ever breaking the failover walk.

        Sync and async hooks are both supported; any exception the hook
        raises is logged and swallowed — a notification failure must not
        influence the fallback semantics in any way.
        """
        try:
            outcome = hook(*args)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as e:
            logger.warning(f"Failover: {hook_name} notification hook failed: {e}")

    async def _get_instances(
        self, state: _WalkState, primary_provider: "Provider", primary_key: str
    ) -> dict[str, "Provider"]:
        """Return the chat-ready provider instances, built lazily on first use.

        Chat-ready provider instances for this user. Built LAZILY: a fully
        successful request never instantiates extra providers nor touches
        the cooldown store — the ladder (and the model enumeration behind
        it) is resolved only when the primary fails.
        """
        if state.instances is None:
            state.instances = self._chat_ready_instances()
            # The primary is always included so the first attempt never
            # depends on registry filtering.
            state.instances.setdefault(primary_key, primary_provider)
        return state.instances

    async def _attempt(
        self,
        state: _WalkState,
        role: Role,
        call: ChatCall,
        pair: FailoverPair,
        provider_instance: "Provider",
    ) -> bool:
        """Run one (provider, model) pair. True on success."""
        key = (pair.provider.lower(), pair.model)
        if key in state.tried:
            return False
        state.tried.add(key)
        if state.first_attempted is None:
            state.first_attempted = pair
        try:
            response, new_messages = await call(provider_instance, pair.model)
        except Exception as e:
            kind = classify_exception(e)
            state.result.attempts.append(
                FailoverAttempt(provider=pair.provider, model=pair.model, ok=False, kind=kind, error=str(e))
            )
            if kind is None:
                raise  # not failover-worthy (context overflow, auth, ...) — propagate honestly
            state.last_exception = e
            state.last_trigger = FailoverTrigger(kind=kind, provider=pair.provider, model=pair.model, original=e)
            await self.cooldown.mark(pair.provider)
            logger.warning(
                f"Failover[{role}]: {pair.provider}/{pair.model or '<default>'} failed ({kind}); "
                f"provider put on a {COOLDOWN_TTL_SECONDS // 60}-minute cooldown."
            )
            return False
        state.result.response = response
        state.result.new_messages = new_messages
        state.result.serving_provider = response.provider or pair.provider
        state.result.serving_model = pair.model or response.model
        # Any successful pair other than the first attempted one (the
        # primary) is by definition a fallback.
        state.result.fallback_used = pair != state.first_attempted
        state.result.attempts.append(FailoverAttempt(provider=pair.provider, model=pair.model, ok=True))
        return True

    async def _walk_pairs(
        self,
        state: _WalkState,
        role: Role,
        pairs_iter: list[FailoverPair],
        instances_map: dict[str, "Provider"],
        cooldown_check: bool,
        primary_key: str,
        on_fallback: FallbackHook | None,
        call: ChatCall,
    ) -> None:
        """Walk the fallback steps in order, attempting each viable pair.

        Stops as soon as a pair succeeds (the result is already recorded in
        the walk state); otherwise returns with the state left for the
        honest-failure epilogue.
        """
        for index, pair in enumerate(pairs_iter):
            instance = instances_map.get(pair.provider.lower())
            if instance is None:
                logger.warning(
                    f"Failover[{role}]: chain step {pair.provider}/{pair.model or '<default>'} "
                    f"is not available to this user — skipping."
                )
                continue
            # No revisiting already-tried pairs within one request.
            if (pair.provider.lower(), pair.model) in state.tried:
                continue
            # Notification (failover_notification task): every step after a
            # failure is a fallback transition — warn BEFORE attempting it.
            # Skipped pairs (cooldown / unavailable instance) are NOT
            # announced: nothing is actually attempted there.
            if state.last_trigger is not None and on_fallback is not None:
                await self._notify("on_fallback", on_fallback, state.last_trigger, pair)
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
            if await self._attempt(state, role, call, pair, instance):
                return

    async def _fail_honestly(self, state: _WalkState, on_failure: FailureHook | None) -> FailoverResult:
        """Finish the walk when every step failed: notify, then raise honestly."""
        # Nothing worked — fail honestly with the original exception.
        if state.result.response is not None:  # pragma: no cover — defensive
            return state.result
        # Notification (failover_notification task): announce the final
        # honest failure, but ONLY when fallbacks were genuinely attempted
        # (attempts beyond the primary). A single-attempt failure with no
        # fallback candidates never fires on_failure — the standard error
        # handling already reports it and "no fallback succeeded" would be
        # noise about fallbacks that never existed.
        if on_failure is not None and state.last_trigger is not None and len(state.result.attempts) > 1:
            fallback_targets = [FailoverPair(provider=a.provider, model=a.model) for a in state.result.attempts[1:]]
            await self._notify("on_failure", on_failure, state.last_trigger, fallback_targets)
        raise (
            state.last_exception or state.last_trigger or RuntimeError("Failover: no provider could serve the request")
        )

    async def run(
        self,
        role: Role,
        primary_provider: "Provider",
        model: str | None,
        call: ChatCall,
        on_fallback: FallbackHook | None = None,
        on_failure: FailureHook | None = None,
    ) -> FailoverResult:
        """Execute ``call`` behind the failover policy for ``role``.

        Args:
            role: ``"master"`` or ``"subagent"``.
            primary_provider: The active provider resolved by the caller.
            model: The active model (None = provider default).
            call: Callable receiving (provider, model) and returning the
                provider-call coroutine with the caller's fixed arguments.
            on_fallback: Optional notification hook invoked (with the
                trigger that fired and the pair about to be attempted)
                immediately before EVERY fallback attempt — auto ladder and
                manual chain alike. Never fired for the primary attempt.
                Hook failures are logged and swallowed: a notification can
                neither block nor break the walk.
            on_failure: Optional notification hook invoked once (with the
                last trigger and the fallback pairs that were ACTUALLY
                attempted) right before the final honest error is raised.
                Only fired when at least one fallback attempt really
                happened — a single-attempt failure (no fallback candidates,
                or DISABLED mode) stays silent here.

        Returns:
            A ``FailoverResult`` (response + fallback metadata).

        Raises:
            Exception: The last provider exception when every step of the
                chain fails (honest failure) — or immediately when the
                exception is not a failover trigger.
        """
        policy = resolve_failover_policy(role)
        state = _WalkState(result=FailoverResult(response=None, new_messages=[]))  # type: ignore[arg-type]
        # Guard against non-string provider names (e.g. AsyncMock doubles in
        # tests): every real provider defines ``name`` as a class-level str.
        primary_name = primary_provider.name if isinstance(primary_provider.name, str) else "primary"
        primary_key = primary_name.lower()

        if policy.mode == FailoverMode.DISABLED:
            await self._attempt(state, role, call, FailoverPair(provider=primary_name, model=model), primary_provider)
            if state.result.response is None:
                raise state.last_exception or state.last_trigger or RuntimeError("Failover: attempt produced no result")
            return state.result

        if policy.mode == FailoverMode.MANUAL:
            # Head of chain = the caller-resolved primary (the user's active
            # selection); the configured chain lists the FALLBACK steps in
            # order. The chain fully replaces auto and is never blocked by
            # the cooldown. An empty configured chain degrades to a single
            # attempt on the primary (same walk as DISABLED, minus the
            # honest-error framing). The tried-set deduplicates a primary
            # that also appears in the configured chain.
            pairs_iter: list[FailoverPair] = [FailoverPair(provider=primary_name, model=model), *policy.chain]
            cooldown_check = False
            # Extra providers are only needed when the chain has fallback
            # steps; an empty chain must not pay for instantiation.
            instances_map = (
                await self._get_instances(state, primary_provider, primary_key)
                if policy.chain
                else {primary_key: primary_provider}
            )
        else:  # AUTO
            # Primary attempt FIRST: the auto ladder — including provider
            # model enumeration — is resolved lazily, only when the primary
            # actually fails. A successful request pays for neither the
            # cooldown store nor the model registry.
            if await self._attempt(
                state, role, call, FailoverPair(provider=primary_name, model=model), primary_provider
            ):
                return state.result
            instances_map = await self._get_instances(state, primary_provider, primary_key)
            blocked: dict[str, bool] = {}
            pairs_iter = await self._build_auto_ladder(
                primary_provider=primary_provider, model=model, instances=instances_map, blocked=blocked
            )
            cooldown_check = True

        await self._walk_pairs(state, role, pairs_iter, instances_map, cooldown_check, primary_key, on_fallback, call)
        # Nothing worked — fail honestly with the original exception.
        return await self._fail_honestly(state, on_failure)


async def run_with_failover(
    role: Role,
    user: "User",
    primary_provider: "Provider",
    model: str | None,
    call: ChatCall,
    on_fallback: FallbackHook | None = None,
    on_failure: FailureHook | None = None,
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
        on_fallback: Optional notification hook fired before every
            fallback attempt (see :meth:`FailoverEngine.run`).
        on_failure: Optional notification hook fired before the final
            honest error (see :meth:`FailoverEngine.run`).

    Returns:
        The ``FailoverResult`` with the response and fallback metadata.
    """
    return await FailoverEngine(user=user).run(
        role=role,
        primary_provider=primary_provider,
        model=model,
        call=call,
        on_fallback=on_fallback,
        on_failure=on_failure,
    )


# Install-time validation: resolve both roles once at import/config-load
# so malformed FAILOVER_CHAIN_* values are reported (warned and skipped)
# at startup rather than on the first chat request. Never raises.
validate_failover_chains()
