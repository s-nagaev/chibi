"""Bot-service wiring tests: the removed post-turn check and thread-fork cache invalidation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chibi.config  # noqa: F401
import chibi.services.bot as bot_module
from chibi.services.bot import handle_clone_thread

if TYPE_CHECKING:
    from chibi.services.interface import UserInterface


class _FakeThreadInterface:
    """Minimal interface surface handle_clone_thread touches."""

    user_data = "user-1"
    chat_data = "chat-1"
    user_id = 1
    storage_id = 111
    thread_id = 42

    def __init__(self) -> None:
        self.created_thread_id: int | None = None

    async def create_thread(self, name: str) -> int:
        assert self.created_thread_id is None
        self.created_thread_id = 555
        return self.created_thread_id

    async def send_message(self, message: str, **kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_post_turn_summarization_check_is_unwired() -> None:
    """The bot service no longer carries the post-turn proactive check.

    The check's job moved to the pre-call guard (``_enforce_context_budget``)
    and the reactive recovery; the bot module must not import or expose the
    trigger so a reintroduction cannot go unnoticed.
    """
    assert not hasattr(bot_module, "check_history_and_summarize")


@pytest.mark.asyncio
async def test_handle_clone_thread_invalidates_usage_cache_for_destination_thread() -> None:
    """Cloning a thread drops the cached prompt size of the destination key."""
    fake = _FakeThreadInterface()
    user = MagicMock()
    user.thread_messages_map = {}

    with (
        patch("chibi.services.bot.get_chibi_user", new=AsyncMock(return_value=user)),
        patch(
            "chibi.services.bot.clone_thread_messages",
            new=AsyncMock(return_value=7),
        ) as clone_mock,
        patch("chibi.services.bot.UsageCacheStore") as store_cls,
    ):
        interface = cast("UserInterface", fake)
        await handle_clone_thread(interface=interface, args=["Copy"])

    clone_mock.assert_awaited_once_with(storage_id=111, old_thread_id=42, new_thread_id=555)
    store_cls.return_value.invalidate.assert_called_once_with(user_id=111, thread_id=555)
