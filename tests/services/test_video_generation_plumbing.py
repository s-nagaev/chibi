"""Unit tests for the video-generation provider abstraction and active-model plumbing."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from chibi.exceptions import NoProviderSelectedError
from chibi.models import SelectedModel, User
from chibi.schemas.app import ModelChangeSchema, VideoResult
from chibi.services import user as user_service
from chibi.services.providers.provider import Provider, RegisteredProviders
from chibi.storage.local import LocalStorage


def _make_dummy_provider(name: str, video_ready: bool) -> type:
    """Plain (non-Provider) stand-in class.

    Must NOT subclass Provider: ``Provider.__init_subclass__`` auto-registers
    any subclass carrying a ``name`` into the global registry, which would
    leak into unrelated tests.
    """

    class _Dummy:
        name: str
        api_key: str
        default_model: str
        video_generation_ready: bool

        def __init__(self, token: str = "") -> None:
            self.token = token

    _Dummy.name = name
    _Dummy.api_key = "test-key"
    _Dummy.default_model = "test-model"
    _Dummy.video_generation_ready = video_ready
    return _Dummy


class TestVideoResult:
    def test_fields_and_defaults(self) -> None:
        result = VideoResult(video=b"mp4-bytes")
        assert result.video == b"mp4-bytes"
        assert result.thumbnail is None
        assert result.duration is None

    def test_all_fields(self) -> None:
        result = VideoResult(video=b"mp4-bytes", thumbnail=b"jpeg-bytes", duration=5)
        assert result.thumbnail == b"jpeg-bytes"
        assert result.duration == 5


class TestModelChangeSchemaVideoFlag:
    def test_video_generation_defaults_to_false(self) -> None:
        schema = ModelChangeSchema(provider="Alibaba", name="wan2.6-t2v", image_generation=False)
        assert schema.video_generation is False

    def test_video_generation_can_be_set(self) -> None:
        schema = ModelChangeSchema(provider="Alibaba", name="wan2.6-t2v", image_generation=False, video_generation=True)
        assert schema.video_generation is True


class TestProviderBase:
    def test_video_generation_ready_flag_defaults_to_false(self) -> None:
        assert Provider.video_generation_ready is False

    def test_default_video_model_defaults_to_none(self) -> None:
        assert Provider.default_video_model is None

    async def test_get_videos_raises_not_implemented_on_base(self) -> None:
        provider = Provider(token="test-token")
        with pytest.raises(NotImplementedError):
            await provider.get_videos(model="any", prompt="a cat")

    async def test_get_videos_signature_accepts_image_and_duration(self) -> None:
        provider = Provider(token="test-token")
        with pytest.raises(NotImplementedError):
            await provider.get_videos(model="any", prompt="a cat", image=b"img", duration=5)


class TestRegisteredProvidersVideo:
    def test_no_video_ready_providers_means_empty_registry(self) -> None:
        chat_provider = _make_dummy_provider("OnlyChat", video_ready=False)
        with patch.object(RegisteredProviders, "available", {"onlychat": chat_provider}):
            registry = RegisteredProviders()
            assert registry.video_generation_ready == {}
            assert registry.first_video_generation_ready is None

    def test_video_ready_provider_is_listed(self) -> None:
        video_provider = _make_dummy_provider("VideoReady", video_ready=True)
        chat_provider = _make_dummy_provider("OnlyChat", video_ready=False)
        with patch.object(RegisteredProviders, "available", {"videoready": video_provider, "onlychat": chat_provider}):
            registry = RegisteredProviders()
            assert set(registry.video_generation_ready) == {"videoready"}
            instance = registry.first_video_generation_ready
            assert instance is not None
            assert instance.name == "VideoReady"


class TestSetActiveModelVideo:
    async def test_video_model_is_routed_to_video_slot(self, tmp_path) -> None:
        db = LocalStorage(storage_path=str(tmp_path))
        interface = MagicMock()
        interface.storage_id = 1
        interface.thread_id = 5

        model = ModelChangeSchema(provider="Alibaba", name="wan2.6-t2v", image_generation=False, video_generation=True)
        await user_service.set_active_model.__wrapped__(db, interface=interface, model=model)

        user = await db.get_or_create_user(user_id=1)
        assert user.thread_selected_video_model[5] == SelectedModel(name="wan2.6-t2v", provider_name="Alibaba")
        assert 5 not in user.thread_selected_image_model
        assert 5 not in user.thread_selected_llm

    async def test_image_model_routing_is_unchanged(self, tmp_path) -> None:
        db = LocalStorage(storage_path=str(tmp_path))
        interface = MagicMock()
        interface.storage_id = 2
        interface.thread_id = 7

        model = ModelChangeSchema(provider="Grok", name="grok-2-image", image_generation=True)
        await user_service.set_active_model.__wrapped__(db, interface=interface, model=model)

        user = await db.get_or_create_user(user_id=2)
        assert user.thread_selected_image_model[7] == SelectedModel(name="grok-2-image", provider_name="Grok")
        assert 7 not in user.thread_selected_video_model


class TestGetActiveVideoProvider:
    def _user_with_providers(self, providers_mock: MagicMock) -> User:
        user = User(id=1)
        return user

    async def test_thread_selection_wins(self) -> None:
        sentinel_provider = MagicMock()
        sentinel_provider.name = "Alibaba"
        providers_mock = MagicMock()
        providers_mock.get.return_value = sentinel_provider

        user = User(id=1, thread_selected_video_model={1: SelectedModel(name="wan2.6-t2v", provider_name="Alibaba")})
        with patch.object(User, "providers", new_callable=PropertyMock, return_value=providers_mock):
            provider = user.get_active_video_provider(thread_id=1)

        assert provider is sentinel_provider
        providers_mock.get.assert_called_once_with(provider_name="Alibaba")

    async def test_falls_back_to_first_video_ready_provider(self) -> None:
        first_ready = MagicMock()
        first_ready.name = "MiniMax"
        sentinel_provider = MagicMock()
        providers_mock = MagicMock()
        providers_mock.first_video_generation_ready = first_ready
        providers_mock.get.return_value = sentinel_provider

        user = User(id=1)
        with patch.object(User, "providers", new_callable=PropertyMock, return_value=providers_mock):
            provider = user.get_active_video_provider(thread_id=1)

        assert provider is sentinel_provider
        providers_mock.get.assert_called_once_with(provider_name="MiniMax")

    async def test_raises_when_no_video_provider_available(self) -> None:
        providers_mock = MagicMock()
        providers_mock.first_video_generation_ready = None

        user = User(id=1)
        with patch.object(User, "providers", new_callable=PropertyMock, return_value=providers_mock):
            with pytest.raises(NoProviderSelectedError):
                user.get_active_video_provider(thread_id=1)

    def test_get_active_video_model_returns_thread_selection(self) -> None:
        user = User(id=1, thread_selected_video_model={3: SelectedModel(name="cogvideox-3", provider_name="ZhipuAI")})
        assert user.get_active_video_model(thread_id=3) == "cogvideox-3"
        assert user.get_active_video_model(thread_id=4) is None

    def test_new_video_fields_have_backward_compatible_defaults(self) -> None:
        user = User(id=1)
        assert user.selected_video_provider_name is None
        assert user.thread_selected_video_model == {}


class TestGetModelsAvailableVideo:
    async def test_video_models_marked_with_active_default(self) -> None:
        provider = MagicMock()
        provider.name = "Alibaba"
        provider.default_video_model = "wan2.6-t2v"

        user = MagicMock()
        user.get_active_video_provider.return_value = provider
        user.get_active_video_model.return_value = None

        db = MagicMock()
        db.get_or_create_user = AsyncMock(return_value=user)

        models = [
            ModelChangeSchema(provider="Alibaba", name="wan2.6-t2v", image_generation=False, video_generation=True),
            ModelChangeSchema(provider="Alibaba", name="wan2.6-t2i", image_generation=True, video_generation=False),
        ]

        cached_mock = AsyncMock(return_value=models)
        with patch("chibi.services.user.get_user_cached_models", cached_mock):
            available = await user_service.get_models_available.__wrapped__(
                db, user_id=1, video_generation=True, thread_id=0
            )

        cached_mock.assert_awaited_once_with(user_id=1, image_generation=False, video_generation=True)
        marked = [m.name for m in available if m.display_name.startswith("\U0001f7e2")]
        assert marked == ["wan2.6-t2v"]

    async def test_video_models_mark_thread_selected_model(self) -> None:
        provider = MagicMock()
        provider.name = "Alibaba"
        provider.default_video_model = "wan2.6-t2v"

        user = MagicMock()
        user.get_active_video_provider.return_value = provider
        user.get_active_video_model.return_value = "wan2.6-t2v-plus"

        db = MagicMock()
        db.get_or_create_user = AsyncMock(return_value=user)

        models = [
            ModelChangeSchema(provider="Alibaba", name="wan2.6-t2v", image_generation=False, video_generation=True),
            ModelChangeSchema(
                provider="Alibaba", name="wan2.6-t2v-plus", image_generation=False, video_generation=True
            ),
        ]

        with patch("chibi.services.user.get_user_cached_models", AsyncMock(return_value=models)):
            available = await user_service.get_models_available.__wrapped__(
                db, user_id=1, video_generation=True, thread_id=0
            )

        marked = [m.name for m in available if m.display_name.startswith("\U0001f7e2")]
        assert marked == ["wan2.6-t2v-plus"]


class TestFilterVideoModels:
    def test_video_generation_filter_keeps_only_video_models(self) -> None:
        provider = Provider(token="test-token")
        models = [
            ModelChangeSchema(provider="P", name="chat-model", image_generation=False, video_generation=False),
            ModelChangeSchema(provider="P", name="video-model", image_generation=False, video_generation=True),
        ]
        filtered = provider.filter_and_return_list_of_models(models=models, video_generation=True)
        assert [m.name for m in filtered] == ["video-model"]
