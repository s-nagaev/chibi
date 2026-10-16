"""Tests for the import-time registration gate of the binary-media generation tools.

Per the approved plan (decision R8), registration is decided when the tool
modules are imported, so the registration matrix is exercised through module
reload (``importlib.reload``) with the settings adjusted before each reload.
The ``RegisteredChibiTools`` registry snapshot fixture restores the original
registry and settings afterwards so later tests are unaffected.
"""

import importlib
from collections.abc import Generator
from typing import Any

import pytest
from loguru import logger

from chibi.config import application_settings, gpt_settings
from chibi.config.app import ClientType
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools import utils as tools_utils
from chibi.services.providers.tools.utils import MEDIA_GENERATION_TOOL_NAMES, media_generation_tools_enabled

MEDIA_TOOL_MODULES = ("chibi.services.providers.tools.media", "chibi.services.providers.tools.video")
LISTING_TOOL_NAMES = ("get_available_image_generation_models", "get_available_video_generation_models")


@pytest.fixture
def registry_snapshot() -> Generator[dict[str, Any], None, None]:
    """Snapshot the tool registry, settings and gate state; restore all after the test."""
    snapshot = dict(RegisteredChibiTools.tools_map)
    original_client = application_settings.client
    original_fs = gpt_settings.filesystem_access
    original_suno_key = gpt_settings.suno_key
    original_elevenlabs_key = gpt_settings.elevenlabs_api_key
    original_notice_flag = tools_utils._media_gate_notice_logged
    yield snapshot
    RegisteredChibiTools.tools_map.clear()
    RegisteredChibiTools.tools_map.update(snapshot)
    application_settings.client = original_client
    gpt_settings.filesystem_access = original_fs
    gpt_settings.suno_key = original_suno_key
    gpt_settings.elevenlabs_api_key = original_elevenlabs_key
    tools_utils._media_gate_notice_logged = original_notice_flag
    for module_name in MEDIA_TOOL_MODULES:
        importlib.reload(importlib.import_module(module_name))


def reload_media_tools() -> None:
    """Re-import the tool modules so the import-time gates re-evaluate.

    The six media tool names are deregistered first: a class whose fresh
    ``register`` evaluates to False does not remove its stale registry entry
    from a previous import, and the assertion target is the decision of this
    reload, not leftovers.
    """
    RegisteredChibiTools.deregister_tools(list(MEDIA_GENERATION_TOOL_NAMES))
    for module_name in MEDIA_TOOL_MODULES:
        importlib.reload(importlib.import_module(module_name))


def registered_media_tools() -> set[str]:
    """Return the subset of media generation tools currently registered."""
    return {name for name in MEDIA_GENERATION_TOOL_NAMES if name in RegisteredChibiTools.tools_map}


class TestGateFunction:
    """The client→capability gate itself, without reload."""

    @pytest.mark.parametrize(
        ("client", "filesystem_access", "expected"),
        [
            ("telegram", False, True),
            ("telegram", True, True),
            ("terminal", True, True),
            ("terminal", False, False),
            ("tui", False, False),
            ("vscode", False, False),
            ("neovim", False, False),
        ],
    )
    def test_matrix(
        self,
        registry_snapshot: dict[str, Any],
        client: ClientType,
        filesystem_access: bool,
        expected: bool,
    ) -> None:
        application_settings.client = client
        gpt_settings.filesystem_access = filesystem_access

        assert media_generation_tools_enabled() is expected


class TestRegistrationMatrix:
    """The six media tools register exactly per the runner × filesystem matrix."""

    def test_telegram_with_filesystem_disabled_registers(self, registry_snapshot: dict[str, Any]) -> None:
        application_settings.client = "telegram"
        gpt_settings.filesystem_access = False
        gpt_settings.suno_key = "key"
        gpt_settings.elevenlabs_api_key = "key"

        reload_media_tools()

        assert registered_media_tools() == set(MEDIA_GENERATION_TOOL_NAMES)

    def test_telegram_with_filesystem_enabled_registers(self, registry_snapshot: dict[str, Any]) -> None:
        application_settings.client = "telegram"
        gpt_settings.filesystem_access = True
        gpt_settings.suno_key = "key"
        gpt_settings.elevenlabs_api_key = "key"

        reload_media_tools()

        assert registered_media_tools() == set(MEDIA_GENERATION_TOOL_NAMES)

    def test_non_delivering_runner_with_filesystem_enabled_registers(self, registry_snapshot: dict[str, Any]) -> None:
        application_settings.client = "terminal"
        gpt_settings.filesystem_access = True
        gpt_settings.suno_key = "key"
        gpt_settings.elevenlabs_api_key = "key"

        reload_media_tools()

        assert registered_media_tools() == set(MEDIA_GENERATION_TOOL_NAMES)

    def test_non_delivering_runner_with_filesystem_disabled_does_not_register(
        self, registry_snapshot: dict[str, Any]
    ) -> None:
        application_settings.client = "terminal"
        gpt_settings.filesystem_access = False
        gpt_settings.suno_key = "key"
        gpt_settings.elevenlabs_api_key = "key"

        reload_media_tools()

        assert registered_media_tools() == set()
        assert all(name in RegisteredChibiTools.tools_map for name in LISTING_TOOL_NAMES)

    def test_provider_key_still_gates_vendor_tools(self, registry_snapshot: dict[str, Any]) -> None:
        application_settings.client = "telegram"
        gpt_settings.filesystem_access = True
        gpt_settings.suno_key = None
        gpt_settings.elevenlabs_api_key = None

        reload_media_tools()

        assert registered_media_tools() == {"generate_image", "generate_video", "text_to_speech"}


class TestTerminalImportTrap:
    """The terminal REPL imports the tool modules under the "telegram" default."""

    def test_import_time_registration_traps_then_startup_deregisters(self, registry_snapshot: dict[str, Any]) -> None:
        from chibi.runners.terminal import _deregister_terminal_incompatible_tools

        application_settings.client = "telegram"
        gpt_settings.filesystem_access = False
        gpt_settings.suno_key = "key"
        gpt_settings.elevenlabs_api_key = "key"

        reload_media_tools()
        assert registered_media_tools() == set(MEDIA_GENERATION_TOOL_NAMES)

        application_settings.client = "terminal"
        _deregister_terminal_incompatible_tools()

        assert registered_media_tools() == set()

    def test_terminal_startup_keeps_tools_with_filesystem_access(self, registry_snapshot: dict[str, Any]) -> None:
        from chibi.runners.terminal import _deregister_terminal_incompatible_tools

        application_settings.client = "telegram"
        gpt_settings.filesystem_access = True
        gpt_settings.suno_key = "key"
        gpt_settings.elevenlabs_api_key = "key"

        reload_media_tools()
        assert registered_media_tools() == set(MEDIA_GENERATION_TOOL_NAMES)

        application_settings.client = "terminal"
        _deregister_terminal_incompatible_tools()

        assert registered_media_tools() == set(MEDIA_GENERATION_TOOL_NAMES)


class TestStartupNotice:
    """The gate explains itself when the media tools are disabled."""

    def test_gated_off_logs_warning_naming_the_setting(self, registry_snapshot: dict[str, Any]) -> None:
        application_settings.client = "terminal"
        gpt_settings.filesystem_access = False
        tools_utils._media_gate_notice_logged = False
        notices: list[str] = []
        handler_id = logger.add(notices.append, level="WARNING", format="{message}")
        try:
            assert media_generation_tools_enabled() is False
        finally:
            logger.remove(handler_id)

        assert any("FILESYSTEM_ACCESS" in message for message in notices)

    def test_notice_logged_once_for_repeated_gate_checks(self, registry_snapshot: dict[str, Any]) -> None:
        application_settings.client = "terminal"
        gpt_settings.filesystem_access = False
        tools_utils._media_gate_notice_logged = False
        notices: list[str] = []
        handler_id = logger.add(notices.append, level="WARNING", format="{message}")
        try:
            assert media_generation_tools_enabled() is False
            media_generation_tools_enabled()
            media_generation_tools_enabled()
        finally:
            logger.remove(handler_id)

        assert sum("FILESYSTEM_ACCESS" in message for message in notices) == 1
