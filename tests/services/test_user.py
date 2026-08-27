"""Tests for chibi.services.user module."""

from pathlib import Path

from chibi.services.user import get_cwd, set_thread_working_dir, set_working_dir
from chibi.storage.local import LocalStorage


async def test_set_thread_working_dir_expands_tilde(tmp_path: Path) -> None:
    """Test that the thread-scoped setter normalizes '~' paths and persists them."""
    db = LocalStorage(storage_path=str(tmp_path))

    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=42, new_wd="~/project")

    user = await db.get_or_create_user(user_id=1)
    expected = str(Path("~/project").expanduser())
    assert user.thread_working_dirs[42] == expected
    assert not expected.startswith("~")
    refreshed = await db.get_or_create_user(user_id=1)
    assert refreshed.get_effective_working_dir(thread_id=42) == expected


async def test_set_thread_working_dir_no_existence_check(tmp_path: Path) -> None:
    """Test that a non-existent directory can still be stored."""
    db = LocalStorage(storage_path=str(tmp_path))
    nonexistent = tmp_path / "does_not_exist"

    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=7, new_wd=str(nonexistent))

    user = await db.get_or_create_user(user_id=1)
    assert user.thread_working_dirs[7] == str(nonexistent)


async def test_set_thread_working_dir_overwrites_existing_slot(tmp_path: Path) -> None:
    """Test that setting an override twice updates the slot without touching legacy fields."""
    db = LocalStorage(storage_path=str(tmp_path))
    user = await db.get_or_create_user(user_id=1)
    legacy = user.working_dir

    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=3, new_wd="/first")
    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=3, new_wd="~/second")

    updated = await db.get_or_create_user(user_id=1)
    assert updated.thread_working_dirs[3] == str(Path("~/second").expanduser())
    assert updated.get_effective_working_dir(thread_id=4) == legacy


async def test_legacy_set_working_dir_stores_value_verbatim(tmp_path: Path) -> None:
    """Test that legacy set_working_dir/get_cwd behavior stays unchanged."""
    db = LocalStorage(storage_path=str(tmp_path))

    await set_working_dir.__wrapped__(db, user_id=1, new_wd="~/legacy")

    user = await db.get_or_create_user(user_id=1)
    assert user.working_dir == "~/legacy"
    assert await get_cwd.__wrapped__(db, user_id=1) == "~/legacy"
