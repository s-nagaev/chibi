"""Tests for the single whole-thread history replacement helper and the atomic local write."""

from pathlib import Path

import pytest

from chibi.models import Message
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import COMPACTION_SEED_SOURCE, replace_thread_history
from chibi.storage.local import LocalStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    """Build a real LocalStorage backend rooted in a temp directory."""
    return LocalStorage(storage_path=str(tmp_path / "storage"))


async def _seed_history(storage: LocalStorage, storage_id: int, thread_id: int, contents: list[str]) -> None:
    """Populate a thread with plain user messages.

    Args:
        storage: The local storage backend.
        storage_id: The storage key of the conversation.
        thread_id: The thread to populate.
        contents: Message contents to store, in order.
    """
    user = await storage.get_or_create_user(user_id=storage_id)
    for content in contents:
        message = Message(role="user", content=content)
        await storage.add_message(user=user, message=message, ttl=3600, thread_id=thread_id)


def _make_seed() -> list[Message]:
    """Build a two-message compaction seed carrying the compaction source marker."""
    return [
        Message(role="user", content="What we were talking about?", source=COMPACTION_SEED_SOURCE),
        Message(role="assistant", content="A summary of everything so far.", source=COMPACTION_SEED_SOURCE),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [0, 5])
async def test_replace_leaves_exactly_the_seed(storage: LocalStorage, thread_id: int) -> None:
    """After the replacement the thread holds exactly the seed — nothing appended, nothing kept."""
    storage_id = 1
    await _seed_history(storage, storage_id, thread_id, ["old one", "old two", "old three"])

    seed = _make_seed()
    await replace_thread_history.__wrapped__(storage, storage_id=storage_id, thread_id=thread_id, seed=seed)

    user = await storage.get_or_create_user(user_id=storage_id)
    history = await storage.get_conversation_messages(user=user, thread_id=thread_id)
    assert [(msg.role, msg.content, msg.source) for msg in history] == [
        (msg.role, msg.content, msg.source) for msg in seed
    ]


@pytest.mark.asyncio
async def test_repeated_replace_is_idempotent(storage: LocalStorage) -> None:
    """Replacing twice leaves exactly one seed: idempotent by construction.

    The whole-thread replace wipes and rewrites unconditionally, so the
    result does not depend on the ``Message.source`` marker being read —
    nothing in the codebase reads the marker yet (it exists for future
    recognition by a retry). What this test pins is the property that is
    actually true: two replacements with the same seed leave exactly that
    seed, never four messages.
    """
    storage_id, thread_id = 2, 7
    await _seed_history(storage, storage_id, thread_id, ["old one", "old two"])

    seed = _make_seed()
    await replace_thread_history.__wrapped__(storage, storage_id=storage_id, thread_id=thread_id, seed=seed)
    await replace_thread_history.__wrapped__(storage, storage_id=storage_id, thread_id=thread_id, seed=seed)

    user = await storage.get_or_create_user(user_id=storage_id)
    history = await storage.get_conversation_messages(user=user, thread_id=thread_id)
    assert len(history) == 2
    assert [(msg.role, msg.content, msg.source) for msg in history] == [
        (msg.role, msg.content, msg.source) for msg in seed
    ]


@pytest.mark.asyncio
async def test_replace_with_a_new_seed_swaps_whole_history(storage: LocalStorage) -> None:
    """A second compaction replaces the previous seed: the history is exactly the new seed."""
    storage_id, thread_id = 3, 9
    await _seed_history(storage, storage_id, thread_id, ["old one"])

    first_seed = _make_seed()
    await replace_thread_history.__wrapped__(storage, storage_id=storage_id, thread_id=thread_id, seed=first_seed)

    second_seed = [
        Message(role="user", content="What we were talking about?", source=COMPACTION_SEED_SOURCE),
        Message(role="assistant", content="A newer summary.", source=COMPACTION_SEED_SOURCE),
    ]
    await replace_thread_history.__wrapped__(storage, storage_id=storage_id, thread_id=thread_id, seed=second_seed)

    user = await storage.get_or_create_user(user_id=storage_id)
    history = await storage.get_conversation_messages(user=user, thread_id=thread_id)
    assert [(msg.role, msg.content, msg.source) for msg in history] == [
        (msg.role, msg.content, msg.source) for msg in second_seed
    ]
    assert all("A newer summary." == msg.content for msg in history if msg.role == "assistant")


@pytest.mark.asyncio
async def test_replace_invalidates_usage_cache(storage: LocalStorage) -> None:
    """The single invalidation helper drops the cached prompt size on every replacement."""
    _reset_usage_cache()
    storage_id, thread_id = 4, 11
    await _seed_history(storage, storage_id, thread_id, ["old one"])
    store = UsageCacheStore()
    store._data[store._make_key(user_id=storage_id, thread_id=thread_id)] = 99999

    await replace_thread_history.__wrapped__(storage, storage_id=storage_id, thread_id=thread_id, seed=_make_seed())

    assert store.get(user_id=storage_id, thread_id=thread_id) is None


@pytest.mark.asyncio
async def test_drop_tool_call_history_invalidates_usage_cache(storage: LocalStorage) -> None:
    """clear_tool_call_history (via drop_tool_call_history) drops the cached prompt size too.

    The shared replacement helper routes this path through the same
    invalidation, which is a deliberate behaviour delta: the model-facing
    ``clear_tool_call_history`` tool now additionally invalidates the cached
    context size for the (user, thread). A history replacement must never
    leave a stale cached size behind, and this is the first coverage of
    ``drop_tool_call_history`` in the suite.
    """
    from chibi.services.user import drop_tool_call_history

    _reset_usage_cache()
    storage_id, thread_id = 8, 13
    await _seed_history(storage, storage_id, thread_id, ["turn one", "turn two"])
    store = UsageCacheStore()
    store._data[store._make_key(user_id=storage_id, thread_id=thread_id)] = 99999

    await drop_tool_call_history.__wrapped__(storage, storage_id=storage_id, thread_id=thread_id)

    assert store.get(user_id=storage_id, thread_id=thread_id) is None


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()


@pytest.mark.asyncio
async def test_local_save_user_is_atomic_and_leaves_no_temp_file(storage: LocalStorage) -> None:
    """The local write goes through temp + os.replace: no truncated file, no leftover temp."""
    storage_id = 6
    user = await storage.get_or_create_user(user_id=storage_id)
    user.thread_notes[0] = "note before"

    await storage.save_user(user)

    user.thread_notes[0] = "note after"
    await storage.save_user(user)

    files = list(Path(storage.storage_path).iterdir())
    assert (Path(storage.storage_path) / f"{storage_id}.pkl") in files
    assert all(not path.name.endswith(".tmp") for path in files)

    reloaded = await storage.get_user(user_id=storage_id)
    assert reloaded is not None
    assert reloaded.thread_notes[0] == "note after"
