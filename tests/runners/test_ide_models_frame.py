"""``models`` request / ``models_result`` frame coverage for the IDE stdio transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.config import gpt_settings
from chibi.interfaces.stdio_runner import COMMANDS, PROTOCOL_VERSION, StdioRunner
from chibi.schemas.app import ModelChangeSchema
from chibi.services.user import get_default_chat_model
from chibi.storage.local import LocalStorage

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ide_protocol"

ACTIVE_MARKER = "🟢"
ACTIVE_MARKER_VARIATION_SELECTOR = "\ufe0f"

READY_FRAME: dict[str, Any] = {
    "type": "ready",
    "protocol_version": PROTOCOL_VERSION,
    "server": {"name": "chibi", "version": "1.0.0"},
    "capabilities": {"commands": COMMANDS, "models": True},
}

CANONICAL_MODELS = [
    ModelChangeSchema(provider="openai", name="gpt-example", display_name="GPT-Example", image_generation=False),
    ModelChangeSchema(provider="alibaba", name="qwen-fake", display_name="Qwen-Fake", image_generation=False),
]
IMAGE_MODELS = [
    *CANONICAL_MODELS,
    ModelChangeSchema(provider="openai", name="gpt-imagine", display_name="GPT-Imagine", image_generation=True),
]

CANONICAL_DEFAULT = ("openai", "gpt-example")
BARE_LISTING = "1. 🟢 GPT-Example️ (openai)\n2. Qwen-Fake (alibaba)"
IMAGE_LISTING = "1. 🟢 GPT-Example️ (openai)\n2. Qwen-Fake (alibaba)\n3. GPT-Imagine (openai)"


class StubProvider:
    """Minimal stand-in for a provider instance in the default-resolution chain."""

    def __init__(self, name: str, default_model: str) -> None:
        """Initialize the stub with its wire-visible identity.

        Args:
            name: Provider name.
            default_model: The provider's configured default model.
        """
        self.name = name
        self.default_model = default_model


class StubProviders:
    """Minimal stand-in for the provider registry of a user."""

    def __init__(self, providers: dict[str, StubProvider] | None = None, first: StubProvider | None = None) -> None:
        """Initialize the registry with its resolvable providers.

        Args:
            providers: Providers resolvable by name.
            first: The first chat-ready provider, if any.
        """
        self._providers = providers or {}
        self._first = first

    def get(self, provider_name: str) -> StubProvider | None:
        """Resolve one provider by name.

        Args:
            provider_name: Name of the provider to resolve.

        Returns:
            The stub provider, or None when it is not configured.
        """
        return self._providers.get(provider_name)

    @property
    def first_chat_ready(self) -> StubProvider | None:
        """Return the first chat-ready provider.

        Returns:
            The stub provider, or None when no provider is chat-ready.
        """
        return self._first


def marked(models: list[ModelChangeSchema], active: str | None) -> list[ModelChangeSchema]:
    """Return a listing copy carrying the active-selection marker on one model.

    The marker is applied the way the real listing applies it (prefix plus a
    trailing variation selector) so the frame's marker stripping is exercised
    rather than assumed.

    Args:
        models: Base listing to copy.
        active: Model id to mark as active, or None to mark nothing.

    Returns:
        A detached listing with the marker applied.
    """
    listing = [model.model_copy() for model in models]
    for model in listing:
        if model.name == active:
            model.display_name = f"{ACTIVE_MARKER} {model.display_name}{ACTIVE_MARKER_VARIATION_SELECTOR}"
    return listing


class ListingRegistry:
    """Fake of the model-listing boundary that records the requested thread."""

    def __init__(self) -> None:
        """Initialize the default listing state."""
        self.reset()

    def reset(self) -> None:
        """Return the registry to its default listing and active model."""
        self.threads: list[int] = []
        self.models: list[ModelChangeSchema] = list(CANONICAL_MODELS)
        self.active_by_thread: dict[int, str] = {0: "gpt-example"}

    async def models_available(self, user_id: int, thread_id: int = 0, **kwargs: Any) -> list[ModelChangeSchema]:
        """Record the thread and answer with the configured listing.

        Args:
            user_id: Storage identity the listing is requested for.
            thread_id: Thread the active model is marked for.
            **kwargs: Additional listing flags ignored by the fake.

        Returns:
            The listing with the configured model marked active.
        """
        self.threads.append(thread_id)
        return marked(self.models, self.active_by_thread.get(thread_id))


REGISTRY = ListingRegistry()


def fixture(name: str) -> list[dict[str, Any]]:
    """Load JSONL fixture records.

    Args:
        name: Fixture file name.

    Returns:
        The parsed records, in file order.
    """
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line.strip()]


class OutputRecorder:
    """Capture protocol frames written through the locked line writer."""

    def __init__(self) -> None:
        """Initialize empty captured output."""
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: Emitted JSON-compatible protocol frame.
        """
        self.frames.append(message)


def runner() -> tuple[StdioRunner, list[dict[str, Any]]]:
    """Create a runner with captured protocol output.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = StdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write_line"] = recorder
    return instance, recorder.frames


def initialize() -> dict[str, Any]:
    """Build the canonical handshake message.

    Returns:
        A v1 initialize message.
    """
    return {"type": "initialize", "protocol_version": PROTOCOL_VERSION}


def models_message(**extra: Any) -> dict[str, Any]:
    """Build a ``models`` message with the given extra fields.

    Args:
        **extra: Extra fields merged into the canonical message.

    Returns:
        The protocol message under test.
    """
    message: dict[str, Any] = {"type": "models", "request_id": "01HXY9K5MODELS0"}
    message.update(extra)
    return message


def model_request(request_id: str, thread_id: int = 0, prompt: str = "/model") -> dict[str, Any]:
    """Build a valid request carrying the given prompt.

    Args:
        request_id: Request identifier.
        thread_id: Thread the request runs on.
        prompt: Prompt text.

    Returns:
        A protocol request message.
    """
    return {
        "type": "request",
        "request_id": request_id,
        "thread_id": thread_id,
        "prompt": prompt,
        "workspace_root": "/tmp",
        "active_file": None,
        "selection": None,
        "cursor_position": None,
        "language_id": None,
    }


def models_result(output: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the first ``models_result`` frame of the captured output.

    Args:
        output: Captured frame list.

    Returns:
        The first models_result frame.

    Raises:
        AssertionError: When no models_result frame was emitted.
    """
    for frame in output:
        if frame.get("type") == "models_result":
            return frame
    raise AssertionError(f"Missing models_result frame: {output}")


def errors(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all error frames of the captured output.

    Args:
        output: Captured frame list.

    Returns:
        The error frames in wire order.
    """
    return [frame for frame in output if frame.get("type") == "error"]


async def wait_for_result(output: list[dict[str, Any]], request_id: str) -> dict[str, Any]:
    """Wait until a request has emitted its result frame.

    Args:
        output: Captured frame list.
        request_id: Request whose result frame is awaited.

    Returns:
        The result frame.

    Raises:
        AssertionError: When no result frame arrives.
    """
    for _ in range(200):
        for frame in output:
            if frame.get("type") == "result" and frame.get("request_id") == request_id:
                return frame
        await asyncio.sleep(0.01)
    raise AssertionError(f"Missing result frame for {request_id}: {output}")


def listing_patch(registry: ListingRegistry | None = None) -> Any:
    """Patch the transport's listing boundary with the fake registry.

    Args:
        registry: Registry to bind, defaulting to the shared one.

    Returns:
        The patch context manager.
    """
    active = registry or REGISTRY
    return patch("chibi.interfaces.stdio_runner.get_models_available", new=active.models_available)


def default_patch(selection: tuple[str, str] | None = CANONICAL_DEFAULT) -> Any:
    """Patch the account-level default resolution.

    Args:
        selection: Provider/model pair the resolution reports, or None.

    Returns:
        The patch context manager.
    """
    return patch("chibi.interfaces.stdio_runner.get_default_chat_model", new=AsyncMock(return_value=selection))


@pytest.fixture(autouse=True)
def reset_registry() -> Iterator[None]:
    """Return the shared listing registry to its default state."""
    REGISTRY.reset()
    yield
    REGISTRY.reset()


class TestHandshakeCapability:
    """The ``models`` capability advertised in the ready frame."""

    @pytest.mark.asyncio
    async def test_ready_advertises_models_and_keeps_commands(self) -> None:
        """The ready frame gains the models flag and keeps the commands list."""
        instance, output = runner()

        await instance._handle_message(initialize())

        assert output == [READY_FRAME]
        assert output[0]["capabilities"]["commands"] == COMMANDS


class TestModelsResult:
    """The correlated ``models_result`` answer."""

    @pytest.mark.asyncio
    async def test_canonical_models_scenario_matches_fixtures(self) -> None:
        """Replay the canonical listing scenario and compare every frame."""
        instance, output = runner()

        with listing_patch(), default_patch():
            for message in fixture("valid_models_input.jsonl"):
                await instance._handle_message(message)

        assert output == fixture("valid_models_output.jsonl")

    @pytest.mark.asyncio
    async def test_active_model_is_thread_scoped_and_marker_free(self) -> None:
        """The active flag follows the requested thread and names lose the marker."""
        instance, output = runner()
        REGISTRY.active_by_thread = {7: "qwen-fake"}

        with listing_patch(), default_patch():
            await instance._handle_message(initialize())
            await instance._handle_message(models_message(thread_id=7))

        assert REGISTRY.threads == [7]
        assert models_result(output)["models"] == [
            {
                "name": "gpt-example",
                "provider": "openai",
                "display_name": "GPT-Example",
                "active": False,
                "image_generation": False,
            },
            {
                "name": "qwen-fake",
                "provider": "alibaba",
                "display_name": "Qwen-Fake",
                "active": True,
                "image_generation": False,
            },
        ]

    @pytest.mark.asyncio
    async def test_missing_thread_id_queries_the_default_thread(self) -> None:
        """A frame without thread_id asks the listing about thread 0."""
        instance, output = runner()

        with listing_patch(), default_patch():
            await instance._handle_message(initialize())
            await instance._handle_message(models_message())

        assert REGISTRY.threads == [0]
        assert [entry["active"] for entry in models_result(output)["models"]] == [True, False]

    @pytest.mark.asyncio
    async def test_default_ignores_the_thread_selection(self) -> None:
        """The advertised default is the account selection, not the thread one."""
        instance, output = runner()
        REGISTRY.active_by_thread = {7: "qwen-fake"}

        with listing_patch(), default_patch(("openai", "gpt-example")):
            await instance._handle_message(initialize())
            await instance._handle_message(models_message(thread_id=7))

        result = models_result(output)
        assert result["default"] == {"provider": "openai", "model": "gpt-example"}
        assert next(entry for entry in result["models"] if entry["active"])["name"] == "qwen-fake"

    @pytest.mark.asyncio
    async def test_default_is_null_without_a_chat_provider(self) -> None:
        """No resolvable chat provider leaves the default explicitly null."""
        instance, output = runner()

        with listing_patch(), default_patch(None):
            await instance._handle_message(initialize())
            await instance._handle_message(models_message())

        assert models_result(output)["default"] is None

    @pytest.mark.asyncio
    async def test_listing_failure_answers_an_error_frame(self) -> None:
        """A failing listing answers request_failed instead of an empty list."""
        instance, output = runner()

        async def failing(user_id: int, thread_id: int = 0, **kwargs: Any) -> list[ModelChangeSchema]:
            """Raise the listing failure under test.

            Args:
                user_id: Storage identity the listing is requested for.
                thread_id: Thread the listing is requested for.
                **kwargs: Additional listing flags ignored by the fake.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("listing down")

        with patch("chibi.interfaces.stdio_runner.get_models_available", new=failing), default_patch():
            await instance._handle_message(initialize())
            await instance._handle_message(models_message())

        failure = errors(output)
        assert [frame["code"] for frame in failure] == ["request_failed"]
        assert failure[0]["request_id"] == "01HXY9K5MODELS0"
        assert not [frame for frame in output if frame.get("type") == "models_result"]

    @pytest.mark.asyncio
    async def test_image_generation_models_are_listed_in_order(self) -> None:
        """Image-generation models stay in the frame, listing order preserved."""
        instance, output = runner()
        REGISTRY.models = list(IMAGE_MODELS)

        with listing_patch(), default_patch():
            await instance._handle_message(initialize())
            await instance._handle_message(models_message())

        assert [entry["name"] for entry in models_result(output)["models"]] == [
            "gpt-example",
            "qwen-fake",
            "gpt-imagine",
        ]


class TestModelCommandTextUnchanged:
    """``/model`` keeps answering the shared listing text, byte for byte."""

    @pytest.mark.asyncio
    async def test_bare_model_listing_text_is_unchanged(self) -> None:
        """The canonical /model text stays byte-identical to the pre-change output."""
        instance, output = runner()

        with listing_patch():
            await instance._handle_message(initialize())
            await instance._handle_message(model_request("listing"))
            result = await wait_for_result(output, "listing")

        assert result["content"] == BARE_LISTING

    @pytest.mark.asyncio
    async def test_image_model_listing_text_is_unchanged(self) -> None:
        """The /model text of a listing holding image models is untouched too."""
        instance, output = runner()
        REGISTRY.models = list(IMAGE_MODELS)

        with listing_patch():
            await instance._handle_message(initialize())
            await instance._handle_message(model_request("listing"))
            result = await wait_for_result(output, "listing")

        assert result["content"] == IMAGE_LISTING


class TestModelsValidation:
    """Validation of the ``models`` message itself."""

    @pytest.mark.asyncio
    async def test_models_before_initialize_is_rejected(self) -> None:
        """A models frame before the handshake is a not_initialized error."""
        instance, output = runner()

        await instance._handle_message(models_message())

        assert [frame["code"] for frame in errors(output)] == ["not_initialized"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_id",
        [None, "", 17],
        ids=["missing", "empty", "not-a-string"],
    )
    async def test_models_without_a_valid_request_id_is_rejected(self, request_id: Any) -> None:
        """A models frame without a usable request_id answers malformed_request."""
        instance, output = runner()
        message: dict[str, Any] = {"type": "models"}
        if request_id is not None:
            message["request_id"] = request_id

        await instance._handle_message(initialize())
        await instance._handle_message(message)

        failure = errors(output)
        assert [frame["code"] for frame in failure] == ["malformed_request"]
        assert failure[0]["request_id"] is None
        assert not [frame for frame in output if frame.get("type") == "models_result"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("thread_id", [-1, "1", None], ids=["negative", "string", "null"])
    async def test_models_with_a_bad_thread_id_is_rejected(self, thread_id: Any) -> None:
        """A present thread_id must be a non-negative integer."""
        instance, output = runner()

        with listing_patch(), default_patch():
            await instance._handle_message(initialize())
            await instance._handle_message(models_message(thread_id=thread_id))

        failure = errors(output)
        assert [frame["code"] for frame in failure] == ["malformed_request"]
        assert failure[0]["request_id"] == "01HXY9K5MODELS0"
        assert not [frame for frame in output if frame.get("type") == "models_result"]


class TestDefaultChatModelResolution:
    """The account-level default chain behind ``models_result.default``."""

    @pytest.mark.asyncio
    async def test_configured_default_provider_wins(self, tmp_path: Path) -> None:
        """The configured default provider is preferred over the first chat-ready one."""
        db = LocalStorage(storage_path=str(tmp_path))
        await db.get_or_create_user(user_id=1)
        providers = StubProviders(
            {
                "openai": StubProvider("openai", "gpt-example"),
                "alibaba": StubProvider("alibaba", "qwen-fake"),
            },
            first=StubProvider("alibaba", "qwen-fake"),
        )

        with (
            patch("chibi.models.User.providers", new_callable=PropertyMock, return_value=providers),
            patch.object(gpt_settings, "default_provider", "openai"),
        ):
            assert await get_default_chat_model.__wrapped__(db, user_id=1) == ("openai", "gpt-example")

    @pytest.mark.asyncio
    async def test_first_chat_ready_provider_is_the_fallback(self, tmp_path: Path) -> None:
        """Without a configured default the first chat-ready provider answers."""
        db = LocalStorage(storage_path=str(tmp_path))
        await db.get_or_create_user(user_id=1)
        providers = StubProviders(
            {"alibaba": StubProvider("alibaba", "qwen-fake")}, first=StubProvider("alibaba", "qwen-fake")
        )

        with (
            patch("chibi.models.User.providers", new_callable=PropertyMock, return_value=providers),
            patch.object(gpt_settings, "default_provider", None),
        ):
            assert await get_default_chat_model.__wrapped__(db, user_id=1) == ("alibaba", "qwen-fake")

    @pytest.mark.asyncio
    async def test_unresolvable_default_provider_falls_through(self, tmp_path: Path) -> None:
        """A configured default that carries no API key falls through to the next provider."""
        db = LocalStorage(storage_path=str(tmp_path))
        await db.get_or_create_user(user_id=1)
        providers = StubProviders(
            {"alibaba": StubProvider("alibaba", "qwen-fake")}, first=StubProvider("alibaba", "qwen-fake")
        )

        with (
            patch("chibi.models.User.providers", new_callable=PropertyMock, return_value=providers),
            patch.object(gpt_settings, "default_provider", "openai"),
        ):
            assert await get_default_chat_model.__wrapped__(db, user_id=1) == ("alibaba", "qwen-fake")

    @pytest.mark.asyncio
    async def test_no_chat_provider_resolves_to_none(self, tmp_path: Path) -> None:
        """With no usable provider the resolution reports nothing."""
        db = LocalStorage(storage_path=str(tmp_path))
        await db.get_or_create_user(user_id=1)
        providers = StubProviders({}, first=None)

        with (
            patch("chibi.models.User.providers", new_callable=PropertyMock, return_value=providers),
            patch.object(gpt_settings, "default_provider", None),
        ):
            assert await get_default_chat_model.__wrapped__(db, user_id=1) is None
