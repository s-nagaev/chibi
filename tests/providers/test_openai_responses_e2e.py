"""E2E tests for OpenAI Responses API migration.

These tests hit the real OpenAI API and are skipped by default.
Run manually when verifying the migration works correctly.

Usage:
    export OPENAI_API_KEY=sk-...
    pytest tests/providers/test_openai_responses_e2e.py -v -s

Each test prints diagnostic info including the model used, response text,
and token usage. After migration, the provider will log which API path
(Responses or Chat Completions) was taken.
"""

import os

import pytest

from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.providers.openai import OpenAI

TOKEN = None
pytestmark = pytest.mark.skip(reason="manual only — hits real OpenAI API")


@pytest.fixture
def openai_provider() -> OpenAI:
    """Create an OpenAI provider with a real API key."""
    token = TOKEN or os.environ.get("OPENAI_API_KEY")
    if not token:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return OpenAI(token=token)


@pytest.fixture
def test_user() -> User:
    """Create a minimal test user."""
    return User(id=12345)


@pytest.mark.asyncio
async def test_responses_api_basic_text(openai_provider: OpenAI, test_user: User) -> None:
    """T1: Verify basic text response via Responses API (o3-mini).

    Sends a simple arithmetic question and expects a brief correct answer.
    After migration, this should route through the Responses API.
    """
    messages = [Message(role="user", content="What is 2+2? Answer briefly.")]

    response, new_messages = await openai_provider.get_chat_response(
        messages=messages,
        user=test_user,
        caller_storage_id=test_user.id,
        caller_thread_id=0,
        model="o3-mini",
        system_prompt="",  # skip system prompt to avoid DB dependency in e2e
    )

    assert isinstance(response, ChatResponseSchema)
    assert "o3-mini" in response.model
    assert response.provider == "OpenAI"
    assert response.answer
    assert "4" in response.answer
    assert response.usage is not None
    assert response.usage.total_tokens > 0

    print("\n[TEST] T1: Basic text response")
    print(f"[MODEL] {response.model}")
    print(f"[PROVIDER] {response.provider}")
    print(f"[RESPONSE] {response.answer.strip()}")
    print(f"[USAGE] {response.usage.total_tokens} total tokens")
    print(f"[NEW_MESSAGES] {len(new_messages)}")


@pytest.mark.asyncio
async def test_responses_api_tool_call(openai_provider: OpenAI, test_user: User) -> None:
    """T2: Verify tool call flow via Responses API (o3-mini).

    Asks for the current time, which should trigger the GetCurrentDatetimeTool.
    Asserts that a tool call was made by checking new_messages for a tool result.
    After migration, tool calls should flow through the Responses API.
    """
    messages = [Message(role="user", content="What time is it now?")]

    response, new_messages = await openai_provider.get_chat_response(
        messages=messages,
        user=test_user,
        caller_storage_id=test_user.id,
        caller_thread_id=0,
        model="o3-mini",
        system_prompt="",
    )

    assert isinstance(response, ChatResponseSchema)
    assert response.answer

    # Verify a tool call was made: new_messages should contain a tool result
    tool_results = [msg for msg in new_messages if msg.role == "tool"]
    assert len(tool_results) > 0, "Expected at least one tool result in new_messages"

    # Verify the tool result contains datetime info
    tool_result = tool_results[0]
    assert "datetime_now" in tool_result.content or "datetime" in tool_result.content.lower()

    # Verify the final response mentions time/date
    answer_lower = response.answer.lower()
    time_keywords = ("time", "date", "hour", "minute", "am", "pm")
    assert any(kw in answer_lower for kw in time_keywords), (
        f"Expected response to mention time/date, got: {response.answer}"
    )

    print("\n[TEST] T2: Tool call flow")
    print(f"[MODEL] {response.model}")
    print(f"[PROVIDER] {response.provider}")
    print(f"[RESPONSE] {response.answer.strip()}")
    print(f"[USAGE] {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"[NEW_MESSAGES] {len(new_messages)}")
    print(f"[TOOL_RESULTS] {len(tool_results)}")
    for msg in new_messages:
        snippet = msg.content[:120].replace("\n", " ") if msg.content else ""
        print(f"  - role={msg.role} tool_calls={bool(getattr(msg, 'tool_calls', None))} content={snippet}...")
