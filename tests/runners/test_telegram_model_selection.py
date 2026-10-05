"""Regression tests for the /image_models back-button routing and the model marker."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chibi.config  # noqa: F401 — must import before chibi.constants (circular import when run standalone)
from chibi.constants import UserAction, UserContext
from chibi.runners.telegram import ChibiBot
from chibi.schemas.app import ModelChangeSchema
from chibi.services.user import get_models_available

CHAT_MODEL = ModelChangeSchema(provider="Melious", name="glm-5.3-flash", image_generation=False)
IMAGE_MODEL = ModelChangeSchema(provider="Grok", name="grok-2-image", image_generation=True)


def _make_context(action: UserAction) -> MagicMock:
    context = MagicMock()
    context.user_data = {
        UserContext.ACTION: action,
        UserContext.MAPPED_MODELS: {CHAT_MODEL.name: CHAT_MODEL, IMAGE_MODEL.name: IMAGE_MODEL},
        UserContext.MAPPED_MODELS_GROUPED: {"Melious": [CHAT_MODEL], "Grok": [IMAGE_MODEL]},
    }
    return context


def _make_query() -> MagicMock:
    query = MagicMock()
    query.data = "__back_to_providers__"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.delete_message = AsyncMock()
    return query


class TestBackToProvidersRouting:
    """Tapping '← Back to providers' must stay in the flow it was opened from."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("initial_action", "expected_action"),
        [
            (UserAction.SELECT_IMAGE_MODEL, UserAction.SELECT_IMAGE_MODEL_PROVIDER),
            (UserAction.SELECT_CHAT_MODEL, UserAction.SELECT_MODEL_PROVIDER),
        ],
    )
    async def test_back_button_preserves_flow(self, initial_action: UserAction, expected_action: UserAction) -> None:
        bot = ChibiBot(telegram_token="test-token")
        context = _make_context(action=initial_action)
        query = _make_query()

        await bot._compute_model_selection_action(query=query, update=MagicMock(), context=context)

        assert context.user_data[UserContext.ACTION] == expected_action
        query.edit_message_text.assert_awaited_once()
        query.delete_message.assert_not_awaited()


class TestLlmMenuDefaultMarker:
    """The LLM menu must mark the provider's default chat model, not its image model."""

    @pytest.mark.asyncio
    async def test_llm_menu_marks_chat_default_not_image_model(self) -> None:
        provider = MagicMock()
        provider.name = "Grok"
        provider.default_model = "grok-4.6"
        provider.default_image_model = "grok-2-image"

        user = MagicMock()
        user.get_active_llm_provider.return_value = provider
        user.get_active_llm_model.return_value = None

        db = MagicMock()
        db.get_or_create_user = AsyncMock(return_value=user)

        models = [
            ModelChangeSchema(provider="Grok", name="grok-4.6", image_generation=False),
            ModelChangeSchema(provider="Grok", name="grok-2-image", image_generation=False),
        ]

        with patch("chibi.services.user.get_user_cached_models", AsyncMock(return_value=models)):
            available = await get_models_available.__wrapped__(db, user_id=1, thread_id=0, image_generation=False)

        marked = [m.name for m in available if m.display_name.startswith("\U0001f7e2")]
        assert marked == ["grok-4.6"]
