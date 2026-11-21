"""Per-request model selection and ``/model`` text regression for the IDE stdio transport."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import chibi.config  # noqa: F401
from chibi.interfaces.stdio_interface import StdioInterface
from chibi.interfaces.stdio_runner import PROTOCOL_VERSION, StdioRunner
from chibi.schemas.app import ModelChangeSchema

MODELS = [
    ModelChangeSchema(provider="openai", name="gpt-example", display_name="🟢 GPT-Example", image_generation=False),
    ModelChangeSchema(provider="alibaba", name="qwen-fake", display_name="Qwen-Fake", image_generation=False),
]

DEFAULT_MODEL = ("gpt-example", "openai")

BARE_LISTING = "1. 🟢 GPT-Example (openai)\n2. Qwen-Fake (alibaba)"
CONFIRMATION_ACTIVE = "Selected model: 🟢 GPT-Example (openai)"
CONFIRMATION_QWEN = "Selected model: Qwen-Fake (alibaba)"
UNKNOWN_SELECTION = "Unknown model selection. Available models: 🟢 GPT-Example, Qwen-Fake."
UNKNOWN_SELECTION_EMPTY = "Unknown model selection. Available models: none."

ABSENT: Any = object()


class ModelRegistry:
    """Fake of the model-selection boundaries: listing, activation and answering."""

    def __init__(self) -> None:
        """Initialize an empty registry state."""
        self.reset()

    def reset(self) -> None:
        """Drop all recorded activations and return to the default listing."""
        self.activations: list[tuple[int, str, str]] = []
        self.active: dict[int, tuple[str, str]] = {}
        self.available: list[ModelChangeSchema] = list(MODELS)
        self.prompt_calls = 0

    async def models_available(self, user_id: int, thread_id: int = 0, **kwargs: Any) -> list[ModelChangeSchema]:
        """Return the fake provider listing.

        Args:
            user_id: Storage identity the listing is requested for.
            thread_id: Thread the active model is marked for.
            **kwargs: Additional selection flags ignored by the fake.

        Returns:
            The listing currently configured on the registry.
        """
        return list(self.available)

    async def set_active_model(self, interface: StdioInterface, model: ModelChangeSchema) -> None:
        """Record one activation and remember it for the interface thread.

        Args:
            interface: Request-local interface the selection arrives on.
            model: Model being activated.
        """
        self.activations.append((interface.thread_id, model.name, model.provider))
        self.active[interface.thread_id] = (model.name, model.provider)

    def active_model(self, thread_id: int) -> tuple[str, str]:
        """Return the model the next answer on a thread runs on.

        Args:
            thread_id: Thread the model is resolved for.

        Returns:
            The ``(name, provider)`` pair active on the thread.
        """
        return self.active.get(thread_id, DEFAULT_MODEL)


REGISTRY = ModelRegistry()
RUNNERS: list[StdioRunner] = []


async def fake_prompt(interface: StdioInterface) -> None:
    """Answer with the model the thread currently runs on.

    Args:
        interface: Request-local interface bound to the thread.
    """
    REGISTRY.prompt_calls += 1
    model, provider = REGISTRY.active_model(interface.thread_id)
    interface.response_model = model
    interface.response_provider = provider
    await interface.send_message(f"answered by {model}")


class OutputRecorder:
    """Capture protocol frames written through the locked line writer."""

    def __init__(self) -> None:
        """Initialize empty captured output."""
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: JSON-compatible protocol frame.
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
    """Build a canonical initialize message.

    Returns:
        A protocol initialize message without capabilities.
    """
    return {"type": "initialize", "protocol_version": PROTOCOL_VERSION}


def request(request_id: str, thread_id: int = 7, prompt: str = "hello", model: Any = ABSENT) -> dict[str, Any]:
    """Build a request frame, optionally carrying a per-request model selection.

    Args:
        request_id: Request correlation identifier.
        thread_id: Thread the request runs on.
        prompt: User prompt or slash command.
        model: Value of the optional ``model`` field; the sentinel omits the field.

    Returns:
        A protocol request frame.
    """
    message: dict[str, Any] = {
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
    if model is not ABSENT:
        message["model"] = model
    return message


def patches() -> list[Any]:
    """Patch the model boundaries with the registry fakes.

    Returns:
        Patch objects the caller starts and stops.
    """
    return [
        patch("chibi.interfaces.stdio_runner.handle_user_prompt", fake_prompt),
        patch("chibi.interfaces.stdio_runner.get_models_available", REGISTRY.models_available),
        patch("chibi.interfaces.stdio_runner.set_active_model", REGISTRY.set_active_model),
    ]


async def wait_for_frame(output: list[dict[str, Any]], request_id: str, frame_type: str) -> dict[str, Any]:
    """Wait until a frame of the given type for one request is emitted.

    Args:
        output: Captured frame list.
        request_id: Identifier of the request the frame must belong to.
        frame_type: Frame type to wait for.

    Returns:
        The first matching frame.
    """
    for _ in range(300):
        for frame in output:
            if frame.get("type") == frame_type and frame.get("request_id") == request_id:
                return frame
        await asyncio.sleep(0.01)
    raise AssertionError(f"Missing {frame_type} frame for {request_id}: {output}")


def frames_of(output: list[dict[str, Any]], request_id: str, frame_type: str) -> list[dict[str, Any]]:
    """Return every captured frame of one type for one request.

    Args:
        output: Captured frame list.
        request_id: Identifier of the request the frames must belong to.
        frame_type: Frame type to collect.

    Returns:
        The matching frames in wire order.
    """
    return [frame for frame in output if frame.get("type") == frame_type and frame.get("request_id") == request_id]


@pytest.fixture(autouse=True)
async def clean_state() -> Any:
    """Reset the registry and reap any request task left running by a test."""
    REGISTRY.reset()
    yield
    for instance in RUNNERS:
        leaked = [task for task in instance._tasks.values() if not task.done()]
        for task in leaked:
            task.cancel()
        if leaked:
            await asyncio.gather(*leaked, return_exceptions=True)
    RUNNERS.clear()


class TestRequestModelField:
    """The optional ``model`` field of a request frame."""

    @pytest.mark.asyncio
    async def test_valid_model_id_activates_and_result_reports_it(self) -> None:
        """A valid model id activates the model before the prompt and reaches the result frame."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, model="qwen-fake"))
            result = await wait_for_frame(output, "r1", "result")
        finally:
            for item in active:
                item.stop()

        assert REGISTRY.activations == [(7, "qwen-fake", "alibaba")]
        assert result["model"] == "qwen-fake"
        assert result["provider"] == "alibaba"
        assert result["content"] == "answered by qwen-fake"

    @pytest.mark.asyncio
    async def test_model_index_selects_the_listed_model(self) -> None:
        """The 1-based index of the listing is accepted as a selection."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, model="2"))
            result = await wait_for_frame(output, "r1", "result")
        finally:
            for item in active:
                item.stop()

        assert REGISTRY.activations == [(7, "qwen-fake", "alibaba")]
        assert result["model"] == "qwen-fake"

    @pytest.mark.asyncio
    async def test_absent_model_field_runs_on_the_thread_model(self) -> None:
        """A request without the field behaves exactly like today: no activation."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7))
            result = await wait_for_frame(output, "r1", "result")
        finally:
            for item in active:
                item.stop()

        assert REGISTRY.activations == []
        assert result["model"] == "gpt-example"
        assert result["content"] == "answered by gpt-example"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [None, ""], ids=["null", "empty-string"])
    async def test_nullish_model_field_is_a_no_op(self, model: Any) -> None:
        """A null or empty selection is a no-op, never a per-thread override reset."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, model=model))
            result = await wait_for_frame(output, "r1", "result")
        finally:
            for item in active:
                item.stop()

        assert REGISTRY.activations == []
        assert result["model"] == "gpt-example"

    @pytest.mark.asyncio
    async def test_selection_is_scoped_to_the_request_thread(self) -> None:
        """A selection made on one thread leaves another thread on its own model."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, model="qwen-fake"))
            await instance._handle_message(request("r2", thread_id=8))
            selected = await wait_for_frame(output, "r1", "result")
            untouched = await wait_for_frame(output, "r2", "result")
        finally:
            for item in active:
                item.stop()

        assert REGISTRY.activations == [(7, "qwen-fake", "alibaba")]
        assert selected["model"] == "qwen-fake"
        assert untouched["model"] == "gpt-example"

    @pytest.mark.asyncio
    async def test_unknown_model_id_fails_without_running_the_prompt(self) -> None:
        """An unknown selection reuses the /model invalid_argument frame and skips the prompt.

        ``invalid_argument`` is the backend-internal code, translated by the transport into
        the frontend-facing ``invalid_request`` exactly as the ``/model`` command does.
        """
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, model="bogus"))
            error = await wait_for_frame(output, "r1", "error")
        finally:
            for item in active:
                item.stop()

        assert error["code"] == "invalid_request"
        assert error["message"] == UNKNOWN_SELECTION
        assert REGISTRY.prompt_calls == 0
        assert REGISTRY.activations == []
        assert frames_of(output, "r1", "result") == []

    @pytest.mark.asyncio
    async def test_non_string_model_field_is_a_malformed_request(self) -> None:
        """A non-string selection fails request validation and never reaches the prompt."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, model=42))
            error = await wait_for_frame(output, "r1", "error")
        finally:
            for item in active:
                item.stop()

        assert error["code"] == "malformed_request"
        assert REGISTRY.prompt_calls == 0
        assert REGISTRY.activations == []


class TestSlashModelRegression:
    """``/model`` text behavior stays byte-identical after the helper extraction."""

    @pytest.mark.asyncio
    async def test_bare_model_lists_the_models(self) -> None:
        """Bare /model answers the numbered listing and activates nothing."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, prompt="/model"))
            result = await wait_for_frame(output, "r1", "result")
        finally:
            for item in active:
                item.stop()

        assert result["content"] == BARE_LISTING
        assert REGISTRY.activations == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("selection", "expected"),
        [("1", CONFIRMATION_ACTIVE), ("qwen-fake", CONFIRMATION_QWEN), ("Qwen-Fake", CONFIRMATION_QWEN)],
        ids=["index", "id", "display-name"],
    )
    async def test_model_confirmation_text_is_unchanged(self, selection: str, expected: str) -> None:
        """Index, id and display-name selections answer the same confirmation text as before.

        Args:
            selection: The ``/model`` argument under test.
            expected: The confirmation line captured before the refactor.
        """
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, prompt=f"/model {selection}"))
            result = await wait_for_frame(output, "r1", "result")
        finally:
            for item in active:
                item.stop()

        assert result["content"] == expected
        assert len(REGISTRY.activations) == 1

    @pytest.mark.asyncio
    async def test_unknown_slash_selection_error_is_unchanged(self) -> None:
        """An unknown /model argument answers the unchanged listing error."""
        instance, output = runner()
        RUNNERS.append(instance)
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, prompt="/model bogus"))
            error = await wait_for_frame(output, "r1", "error")
        finally:
            for item in active:
                item.stop()

        assert error["code"] == "invalid_request"
        assert error["message"] == UNKNOWN_SELECTION
        assert REGISTRY.activations == []

    @pytest.mark.asyncio
    async def test_empty_listing_text_is_unchanged(self) -> None:
        """With no models configured the listing and the unknown-selection error stay verbatim."""
        instance, output = runner()
        RUNNERS.append(instance)
        REGISTRY.available = []
        active = patches()
        for item in active:
            item.start()
        try:
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", thread_id=7, prompt="/model"))
            listing = await wait_for_frame(output, "r1", "result")
            await instance._handle_message(request("r2", thread_id=7, prompt="/model 1"))
            error = await wait_for_frame(output, "r2", "error")
        finally:
            for item in active:
                item.stop()

        assert listing["content"] == "No models available."
        assert error["message"] == UNKNOWN_SELECTION_EMPTY
