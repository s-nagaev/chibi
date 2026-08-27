"""Tests for the EditorContextProvider protocol and IDEInterface accessor."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import IDEInterface
from chibi.services.interface import EditorContextProvider, TelegramInterface, UserInterface


def test_ide_interface_is_editor_context_provider() -> None:
    """IDEInterface instances satisfy the EditorContextProvider protocol."""
    interface = IDEInterface(1, "hello", {"active_file": "foo.py"}, lambda _: None)

    assert isinstance(interface, EditorContextProvider)


def test_telegram_interface_is_not_editor_context_provider() -> None:
    """TelegramInterface instances do not satisfy the EditorContextProvider protocol."""
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1, type="private"),
        effective_user=SimpleNamespace(id=1, name="test"),
        effective_message=None,
    )
    context = MagicMock()
    interface = TelegramInterface(update, context)  # type: ignore[arg-type]

    assert not isinstance(interface, EditorContextProvider)


def test_user_interface_abc_is_not_editor_context_provider() -> None:
    """The base UserInterface ABC does not implement EditorContextProvider."""

    class MinimalUserInterface(UserInterface):
        pass

    assert not isinstance(MinimalUserInterface(), EditorContextProvider)


def test_ide_interface_editor_context_returns_supplied_dict() -> None:
    """IDEInterface.editor_context returns the dict passed at construction."""
    context = {
        "active_file": "foo.py",
        "selection": {"start_line": 1, "end_line": 2, "text": "bar"},
        "language_id": "python",
        "workspace_root": "/tmp",
        "cursor_position": None,
    }
    interface = IDEInterface(1, "explain", context, lambda _: None)

    assert interface.editor_context == context


def test_ide_interface_context_emits_deprecation_warning() -> None:
    """Accessing IDEInterface.context emits a DeprecationWarning and delegates."""
    context = {"active_file": "foo.py"}
    interface = IDEInterface(1, "hello", context, lambda _: None)

    with pytest.warns(DeprecationWarning, match="IDEInterface.context is deprecated"):
        result = interface.context

    assert result == context


def test_ide_interface_context_warns_via_warnings_warn() -> None:
    """IDEInterface.context calls warnings.warn with the expected arguments."""
    context = {"active_file": "foo.py"}
    interface = IDEInterface(1, "hello", context, lambda _: None)

    with patch("chibi.runners.ide_transport.warnings.warn") as warn_mock:
        result = interface.context

    warn_mock.assert_called_once_with(
        "IDEInterface.context is deprecated; use IDEInterface.editor_context instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    assert result == context
