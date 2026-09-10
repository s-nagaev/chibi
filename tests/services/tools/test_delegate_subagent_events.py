"""DelegateTool subagent lifecycle counting with a fake subagent runner."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import IDEStdioRunner
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers.tools.common import DelegateTool
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.subagent_events import subagent_tracker

RUNNERS: list[IDEStdioRunner] = []


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


def counting_runner() -> tuple[IDEStdioRunner, list[dict[str, Any]]]:
    """Create a runner wired as the tracker sink with captured output.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = IDEStdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write_line"] = recorder
    instance._subagent_events_enabled = True
    subagent_tracker.set_sink(instance)
    instance.begin_request(42, "r1")
    RUNNERS.append(instance)
    return instance, recorder.frames


async def fake_models_available(user_id: int, **kwargs: Any) -> list[ModelChangeSchema]:
    """Fake available-models list matching the delegation target."""
    return [ModelChangeSchema(provider="prov", name="gpt-x", display_name="GPT-X", image_generation=False)]


async def fake_subagent_ok(**kwargs: Any) -> ChatResponseSchema:
    """Fake subagent runner returning a normal answer."""
    return ChatResponseSchema(answer="delegated answer", provider="prov", model="gpt-x", usage=None)


async def fake_subagent_hanging(**kwargs: Any) -> ChatResponseSchema:
    """Fake subagent runner that never finishes on its own."""
    await asyncio.sleep(30)
    raise AssertionError("unreachable")  # pragma: no cover


async def fake_subagent_boom(**kwargs: Any) -> ChatResponseSchema:
    """Fake subagent runner dying with an unexpected exception."""
    raise ValueError("subagent exploded")


def delegate_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build a valid delegation kwargs payload targeting thread 42.

    Args:
        overrides: Values replacing the defaults.

    Returns:
        The kwargs payload for ``DelegateTool.function``.
    """
    kwargs: dict[str, Any] = {
        "user_id": 1,
        "caller_model": "caller-model",
        "caller_provider": "caller-provider",
        "caller_storage_id": 99,
        "caller_thread_id": 42,
    }
    kwargs.update(overrides)
    return kwargs


def agent_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the agent_event frames from the captured output.

    Args:
        frames: Captured frame list.

    Returns:
        The agent_event frames in wire order.
    """
    return [frame for frame in frames if frame.get("type") == "agent_event"]


@pytest.fixture(autouse=True)
async def subagent_event_cleanup() -> Any:
    """Release the tracker sink and settle emission tasks after each test."""
    yield
    subagent_tracker.set_sink(None)
    for instance in RUNNERS:
        await instance.drain_subagent_events()
    RUNNERS.clear()


class TestDelegateLifecycleEvents:
    """Started and finished emission around the delegation attempt."""

    @pytest.mark.asyncio
    async def test_successful_delegation_emits_started_and_finished(self) -> None:
        """An accepted delegation counts one start and one finish."""
        _instance, output = counting_runner()

        with (
            patch("chibi.services.providers.tools.common.get_sub_agent_response", fake_subagent_ok),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", fake_models_available),
        ):
            result = await DelegateTool.function(
                prompt="do something",
                model_name="gpt-x",
                provider_name="prov",
                **delegate_kwargs(),
            )
            await subagent_tracker.drain()

        assert result == {"response": "delegated answer"}
        assert agent_events(output) == [
            {
                "type": "agent_event",
                "request_id": "r1",
                "event": "started",
                "active": 1,
                "total": 1,
                "name": "gpt-x",
            },
            {
                "type": "agent_event",
                "request_id": "r1",
                "event": "finished",
                "active": 0,
                "total": 1,
                "name": "gpt-x",
            },
        ]

    @pytest.mark.asyncio
    async def test_failed_attempt_emits_nothing(self) -> None:
        """A rejected delegation attempt never counts and never emits."""
        instance, output = counting_runner()

        with pytest.raises(ToolException):
            await DelegateTool.function(
                prompt="do something",
                model_name="gpt-x",
                provider_name=None,
                **delegate_kwargs(),
            )
        await subagent_tracker.drain()

        assert agent_events(output) == []
        assert instance._subagent_requests[42]["total"] == 0
        assert instance._subagent_requests[42]["active"] == 0

    @pytest.mark.asyncio
    async def test_timeout_still_emits_finished(self) -> None:
        """A subagent dying by timeout emits its finished event via cleanup."""
        _instance, output = counting_runner()

        with (
            pytest.raises(ToolException),
            patch("chibi.services.providers.tools.common.get_sub_agent_response", fake_subagent_hanging),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", fake_models_available),
        ):
            await DelegateTool.function(
                prompt="do something",
                model_name="gpt-x",
                provider_name="prov",
                timeout=0.01,
                **delegate_kwargs(),
            )
        await subagent_tracker.drain()

        events = agent_events(output)
        assert [(event["event"], event["active"], event["total"]) for event in events] == [
            ("started", 1, 1),
            ("finished", 0, 1),
        ]

    @pytest.mark.asyncio
    async def test_exception_still_emits_finished(self) -> None:
        """A subagent dying by exception emits its finished event via cleanup."""
        _instance, output = counting_runner()

        with (
            pytest.raises(ValueError),
            patch("chibi.services.providers.tools.common.get_sub_agent_response", fake_subagent_boom),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", fake_models_available),
        ):
            await DelegateTool.function(
                prompt="do something",
                model_name="gpt-x",
                provider_name="prov",
                **delegate_kwargs(),
            )
        await subagent_tracker.drain()

        events = agent_events(output)
        assert [(event["event"], event["active"], event["total"]) for event in events] == [
            ("started", 1, 1),
            ("finished", 0, 1),
        ]

    @pytest.mark.asyncio
    async def test_no_sink_is_a_silent_noop(self) -> None:
        """Without an opted-in session the lifecycle calls are invisible."""
        _instance, output = counting_runner()
        subagent_tracker.set_sink(None)
        output.clear()

        with (
            patch("chibi.services.providers.tools.common.get_sub_agent_response", fake_subagent_ok),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", fake_models_available),
        ):
            await DelegateTool.function(
                prompt="do something",
                model_name="gpt-x",
                provider_name="prov",
                **delegate_kwargs(),
            )
            await subagent_tracker.drain()

        assert agent_events(output) == []


async def fake_subagent_fast(**kwargs: Any) -> ChatResponseSchema:
    """Fake subagent runner finishing shortly after the request ends."""
    await asyncio.sleep(0.01)
    return ChatResponseSchema(answer="delegated answer", provider="prov", model="gpt-x", usage=None)


async def fake_subagent_slow(**kwargs: Any) -> ChatResponseSchema:
    """Fake subagent runner finishing well after its sibling."""
    await asyncio.sleep(0.15)
    return ChatResponseSchema(answer="delegated answer", provider="prov", model="gpt-x", usage=None)


class TestBackgroundDelegationOutlivesRequest:
    """Background delegations whose subagents finish after the request ends."""

    @pytest.mark.asyncio
    async def test_finished_frames_reach_the_wire_after_request_ends(self) -> None:
        """Subagents outliving their request emit their lifecycle frames.

        Mirrors the production background path: the delegation coroutine runs
        as an independent task while the parent request drains, seals its
        counters and retires. The retained state keeps the counters alive, so
        both late natural finishes are emitted and the last one retires the
        state instead of leaving a stale counter on the wire.
        """
        instance, output = counting_runner()

        with (
            patch(
                "chibi.services.providers.tools.common.get_sub_agent_response",
                side_effect=[fake_subagent_fast(), fake_subagent_slow()],
            ),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", fake_models_available),
        ):
            tasks = [
                asyncio.create_task(
                    DelegateTool.function(
                        prompt=f"task {index}",
                        model_name="gpt-x",
                        provider_name="prov",
                        **delegate_kwargs(),
                    )
                )
                for index in range(2)
            ]
            for _ in range(300):
                if len([event for event in agent_events(output) if event["event"] == "started"]) == 2:
                    break
                await asyncio.sleep(0.01)
            await subagent_tracker.drain()
            assert [(event["event"], event["active"], event["total"]) for event in agent_events(output)] == [
                ("started", 1, 1),
                ("started", 2, 2),
            ]

            instance.seal_request(42, "r1")
            instance.end_request(42, "r1")
            assert instance._subagent_requests[42]["sealed"] is True
            assert instance._subagent_requests[42]["active"] == 2

            results = await asyncio.gather(*tasks)
            await subagent_tracker.drain()

        assert results == [{"response": "delegated answer"}, {"response": "delegated answer"}]
        events = agent_events(output)
        assert [(event["event"], event["active"], event["total"]) for event in events] == [
            ("started", 1, 1),
            ("started", 2, 2),
            ("finished", 1, 2),
            ("finished", 0, 2),
        ]
        assert 42 not in instance._subagent_requests
