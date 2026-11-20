"""Tests for chibi.models module."""

import functools
import json

import pytest
from anthropic.types import AssistantMessageParam, ToolResultBlockParam, ToolUseBlockParam, UserMessageParam
from google.genai.types import ContentDict, FunctionCallDict, FunctionResponseDict, PartDict
from mistralai.client import UNSET
from mistralai.client.models import TextChunk, ThinkChunk

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import NoProviderSelectedError
from chibi.models import FunctionSchema, Message, SelectedModel, ToolSchema, User, mistral_content_to_text


class TestMessageGoogleConversion:
    """Test conversion methods for Google AI format."""

    def test_to_google_simple_user_message(self):
        """Test converting a simple user message to Google format."""
        message = Message(role="user", content="Hello, how are you?")
        google_content = message.to_google()

        assert google_content == {"role": "user", "parts": [{"text": "Hello, how are you?"}]}

    def test_to_google_simple_assistant_message(self):
        """Test converting a simple assistant message to Google format."""
        message = Message(role="assistant", content="I'm doing well, thank you!")
        google_content = message.to_google()

        assert google_content == {"role": "model", "parts": [{"text": "I'm doing well, thank you!"}]}

    def test_to_google_assistant_with_tool_calls(self):
        """Test converting assistant message with tool calls to Google format."""
        function = FunctionSchema(name="get_weather", arguments=json.dumps({"location": "San Francisco"}))
        tool = ToolSchema(id="call_123", function=function)

        message = Message(role="assistant", content="Let me check the weather for you.", tool_calls=[tool])
        google_content = message.to_google()

        expected = {
            "role": "model",
            "parts": [
                {"text": "Let me check the weather for you."},
                {
                    "function_call": {
                        "id": "call_123",
                        "name": "get_weather",
                        "args": {"location": "San Francisco"},
                    },
                    "thought_signature": None,
                },
            ],
        }
        assert google_content == expected

    def test_to_google_assistant_with_tool_calls_no_content(self):
        """Test converting assistant message with tool calls but no text content."""
        function = FunctionSchema(name="calculate", arguments=json.dumps({"expression": "2 + 2"}))
        tool = ToolSchema(id="call_456", function=function)

        message = Message(role="assistant", content="", tool_calls=[tool])
        google_content = message.to_google()

        expected = {
            "role": "model",
            "parts": [
                {
                    "function_call": {"id": "call_456", "name": "calculate", "args": {"expression": "2 + 2"}},
                    "thought_signature": None,
                }
            ],
        }
        assert google_content == expected

    def test_to_google_tool_response(self):
        """Test converting tool response message to Google format."""
        message = Message(
            role="tool",
            content="The weather in San Francisco is sunny, 72°F",
            tool_call_id="123",
            tool_name="get_weather",
        )
        google_content = message.to_google()

        expected = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "123",
                        "name": "get_weather",
                        "response": "The weather in San Francisco is sunny, 72°F",
                    }
                }
            ],
        }
        assert google_content == expected

    def test_to_google_empty_message(self):
        """Test converting empty message to Google format."""
        message = Message(role="user", content="")
        google_content = message.to_google()

        assert google_content == {"role": "user", "parts": []}

    def test_from_google_simple_user_message(self):
        """Test converting from Google format to simple user message."""
        google_content = ContentDict(
            role="user",
            parts=[
                PartDict(text="Hello, how are you?"),
            ],
        )

        message = Message.from_google(google_content)

        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert message.tool_calls is None
        assert message.tool_call_id is None

    def test_from_google_simple_assistant_message(self):
        """Test converting from Google format to simple assistant message."""
        google_content = ContentDict(
            role="model",
            parts=[
                PartDict(text="I'm doing well, thank you!"),
            ],
        )
        message = Message.from_google(google_content)

        assert message.role == "assistant"
        assert message.content == "I'm doing well, thank you!"
        assert message.tool_calls is None
        assert message.tool_call_id is None

    def test_from_google_assistant_with_function_call(self):
        """Test converting from Google format assistant message with function call."""
        google_content = {
            "role": "model",
            "parts": [
                {"text": "Let me check the weather for you."},
                {"function_call": {"name": "get_weather", "args": {"location": "San Francisco"}}},
            ],
        }
        message = Message.from_google(google_content)

        assert message.role == "assistant"
        assert message.content == "Let me check the weather for you."
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1

        tool_call = message.tool_calls[0]
        assert tool_call.function.arguments
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"location": "San Francisco"}
        assert tool_call.id.startswith("call_")

    def test_from_google_function_response(self):
        """Test converting from Google format function response."""
        tool_call_id = "some_unique_id"
        google_content = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": tool_call_id,
                        "name": "get_weather",
                        "response": {"content": "The weather in San Francisco is sunny, 72°F"},
                    }
                }
            ],
        }
        message = Message.from_google(google_content)

        assert message.role == "tool"
        assert json.loads(message.content) == {"content": "The weather in San Francisco is sunny, 72°F"}
        assert message.tool_call_id == tool_call_id
        assert message.tool_calls is None

    def test_from_google_empty_parts(self):
        """Test converting from Google format with empty parts."""
        google_content = {"role": "user", "parts": []}
        message = Message.from_google(google_content)

        assert message.role == "user"
        assert message.content == ""
        assert message.tool_calls is None
        assert message.tool_call_id is None

    def test_from_google_no_parts(self):
        """Test converting from Google format with no parts key."""
        google_content = {"role": "user"}
        message = Message.from_google(google_content)

        assert message.role == "user"
        assert message.content == ""
        assert message.tool_calls is None
        assert message.tool_call_id is None

    def test_from_google_no_role(self):
        """Test converting from Google format with no role key."""
        google_content = {"parts": [{"text": "Hello"}]}
        message = Message.from_google(google_content)

        assert message.role == "user"  # Default role
        assert message.content == "Hello"
        assert message.tool_calls is None
        assert message.tool_call_id is None

    def test_roundtrip_conversion_simple_message(self):
        """Test that simple message survives roundtrip conversion."""
        original = Message(role="user", content="Hello world")
        google_content = original.to_google()
        converted = Message.from_google(google_content)

        assert converted.role == original.role
        assert converted.content == original.content
        assert converted.tool_calls == original.tool_calls
        assert converted.tool_call_id == original.tool_call_id

    def test_roundtrip_conversion_assistant_message(self):
        """Test that assistant message survives roundtrip conversion."""
        original = Message(role="assistant", content="I'm doing well")
        google_content = original.to_google()
        converted = Message.from_google(google_content)

        assert converted.role == original.role
        assert converted.content == original.content
        assert converted.tool_calls == original.tool_calls
        assert converted.tool_call_id == original.tool_call_id

    def test_roundtrip_conversion_tool_response(self):
        """Test that tool response survives roundtrip conversion."""
        original = Message(
            role="tool",
            content=json.dumps({"result": {"sky": "clear"}}),
            tool_call_id="some_unique_id",
            tool_name="get_weather",
        )
        google_content = original.to_google()
        converted = Message.from_google(google_content)

        assert converted.role == original.role
        assert converted.content == original.content
        assert converted.tool_call_id == original.tool_call_id
        assert converted.tool_name == original.tool_name
        assert converted.tool_calls is None

    def test_roundtrip_conversion_tool_response_g_c_g(self):
        """Test that tool response survives roundtrip conversion. Google-Chibi-Google"""
        source = ContentDict(
            role="user",
            parts=[
                PartDict(
                    function_response=FunctionResponseDict(
                        id="some_unique_id", name="get_weather", response={"result": {"sky": "clear"}}
                    )
                ),
            ],
        )
        converted = Message.from_google(source)
        restored = converted.to_google()
        assert restored == source

    def test_roundtrip_conversion_tool_call_message_g_c_g(self):
        """Test that tool call survives roundtrip conversion. Google-Chibi-Google"""

        source = ContentDict(
            role="model",
            parts=[
                PartDict(
                    function_call=FunctionCallDict(
                        name="get_weather",
                        args={"location": "San Francisco"},
                        id="some_unique_id",
                    ),
                    thought_signature=b"\x12",
                )
            ],
        )
        converted = Message.from_google(source)
        restored = converted.to_google()
        assert restored == source

    def test_tool_calls_with_empty_arguments(self):
        """Test handling tool calls with empty arguments."""
        function = FunctionSchema(name="ping", arguments=None)
        tool = ToolSchema(id="call_789", function=function)

        message = Message(role="assistant", content="", tool_calls=[tool])
        google_content = message.to_google()

        expected = {
            "role": "model",
            "parts": [
                {
                    "function_call": {"name": "ping", "args": {}, "id": "call_789"},
                    "thought_signature": None,
                }
            ],
        }
        assert google_content == expected

    def test_multiple_tool_calls(self):
        """Test handling multiple tool calls in one message."""
        function1 = FunctionSchema(name="get_weather", arguments=json.dumps({"location": "SF"}))
        function2 = FunctionSchema(name="get_time", arguments=json.dumps({"timezone": "PST"}))
        tool1 = ToolSchema(id="call_1", function=function1)
        tool2 = ToolSchema(id="call_2", function=function2)

        message = Message(role="assistant", content="Getting weather and time", tool_calls=[tool1, tool2])
        google_content = message.to_google()

        assert google_content["role"] == "model"
        assert google_content["parts"]
        assert len(google_content["parts"]) == 3  # 1 text + 2 function calls
        assert google_content["parts"][0]["text"] == "Getting weather and time"
        assert google_content["parts"][1]["function_call"]
        assert google_content["parts"][1]["function_call"]["name"] == "get_weather"
        assert google_content["parts"][2]["function_call"]
        assert google_content["parts"][2]["function_call"]["name"] == "get_time"


class TestMistralContentToText:
    """Test flattening of Mistral message and delta content into plain text."""

    def test_plain_string_returned_verbatim(self) -> None:
        """Test that a plain string content is returned exactly as-is."""
        assert mistral_content_to_text("Hello, world") == "Hello, world"

    def test_list_of_text_chunks_concatenated(self) -> None:
        """Test that a list of text chunks flattens into the concatenated body text."""
        content = [TextChunk(text="Hello, "), TextChunk(text="world")]

        assert mistral_content_to_text(content) == "Hello, world"

    def test_list_of_reasoning_chunks_is_dropped(self) -> None:
        """Test that reasoning (ThinkChunk) content never leaks into the result."""
        reasoning = ThinkChunk(thinking=[TextChunk(text="secret chain of thought")])

        result = mistral_content_to_text([reasoning])

        assert result == ""
        assert "secret chain of thought" not in result

    def test_mixed_list_keeps_only_text_chunks(self) -> None:
        """Test that a mixed list keeps body text and drops the reasoning chunks."""
        reasoning = ThinkChunk(thinking=[TextChunk(text="private reasoning")])
        content = [reasoning, TextChunk(text="The answer"), TextChunk(text=" is 42")]

        assert mistral_content_to_text(content) == "The answer is 42"
        assert "private reasoning" not in mistral_content_to_text(content)

    def test_unset_value_flattens_to_empty_string(self) -> None:
        """Test that unset, None and unexpected values flatten to an empty string."""
        assert mistral_content_to_text(UNSET) == ""
        assert mistral_content_to_text(None) == ""
        assert mistral_content_to_text(42) == ""


class TestMessageAnthropicConversion:
    def test_anthropic_to_chibi_tool_call(self) -> None:
        args = {"max_results": 5, "search_phrase": "weather forecast Krakow February 15 2026"}
        source = AssistantMessageParam(
            role="assistant",
            content=[
                ToolUseBlockParam(id="call_function_npanp5e9uv0s_1", input=args, name="get_weather", type="tool_use")
            ],
        )
        result = Message.from_anthropic(source)
        assert result.role == "assistant"
        assert result.tool_name == "get_weather"
        assert result.tool_call_id == "call_function_npanp5e9uv0s_1"
        assert result.tool_calls

        tool_call = result.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function

        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == json.dumps(args)

    def test_anthropic_to_chibi_tool_result(self) -> None:
        tool_result = {"tool_name": "ddgs_web_search", "status": "ok", "result": "some data here"}

        source = UserMessageParam(
            role="user",
            content=[
                ToolResultBlockParam(
                    tool_use_id="call_function_npanp5e9uv0s_1",
                    type="tool_result",
                    content=json.dumps(tool_result),
                )
            ],
        )
        result = Message.from_anthropic(source)
        assert result.role == "tool"
        assert result.tool_call_id == "call_function_npanp5e9uv0s_1"
        assert result.tool_name == "ddgs_web_search"
        assert result.content == json.dumps(tool_result)


class TestUserEffectiveWorkingDir:
    """Test thread-scoped working directory resolution on User."""

    def test_thread_override_wins(self) -> None:
        """Test that a thread-specific working directory takes priority over the legacy one."""
        user = User(id=1)
        user.working_dir = "/legacy"
        user.thread_working_dirs[42] = "/thread/override"

        assert user.get_effective_working_dir(thread_id=42) == "/thread/override"

    def test_missing_key_falls_back_to_legacy(self) -> None:
        """Test that a thread without an override falls back to the legacy user-level directory."""
        user = User(id=1)
        user.working_dir = "/legacy"

        assert user.get_effective_working_dir(thread_id=42) == "/legacy"

    def test_legacy_empty_falls_back_to_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that an empty legacy directory falls back to the application setting."""
        monkeypatch.setattr(application_settings, "working_dir", "/cfg/default")
        user = User(id=1)
        user.working_dir = ""

        assert user.get_effective_working_dir(thread_id=42) == "/cfg/default"

    def test_thread_id_none_skips_thread_level(self) -> None:
        """Test that thread_id=None skips the thread-level lookup entirely."""
        user = User(id=1)
        user.working_dir = "/legacy"
        user.thread_working_dirs[42] = "/thread/override"

        assert user.get_effective_working_dir(thread_id=None) == "/legacy"

    def test_stored_value_returned_asis_without_expanduser(self) -> None:
        """Test that the stored value is returned verbatim — no expansion on read."""
        user = User(id=1)
        user.thread_working_dirs[42] = "~/relative/path"

        result = user.get_effective_working_dir(thread_id=42)

        assert result == "~/relative/path"

    def test_old_records_without_field_get_default_dict(self) -> None:
        """Test that users constructed from old persisted records without the field get the default."""
        user = User(id=1, working_dir="/legacy")

        assert user.thread_working_dirs == {}
        assert user.get_effective_working_dir(thread_id=1) == "/legacy"


class _FakeProvider:
    """Provider stand-in exposing only what user-level LLM resolution reads."""

    def __init__(self, name: str, default_model: str | None = None) -> None:
        """Create a fake provider.

        Args:
            name: Name the fake registry resolves this provider by.
            default_model: Model the provider reports as its own default.
        """
        self.name = name
        self.default_model = default_model


class _FakeProviderRegistry:
    """RegisteredProviders stand-in limited to the lookups User performs."""

    def __init__(self, providers: list[_FakeProvider], user_api_keys: dict[str, str] | None = None) -> None:
        """Store the providers this registry exposes.

        Args:
            providers: Providers visible to lookups, in registration order.
            user_api_keys: Accepted for signature compatibility with RegisteredProviders.
        """
        self.providers = providers

    def get(self, provider_name: str) -> _FakeProvider | None:
        """Resolve a provider by name.

        Args:
            provider_name: Name to resolve.

        Returns:
            The matching provider, or None when it is not registered.
        """
        for provider in self.providers:
            if provider.name == provider_name.lower():
                return provider
        return None

    @property
    def first_chat_ready(self) -> _FakeProvider | None:
        """Return the first chat-ready provider.

        Returns:
            The first registered provider, or None when the registry is empty.
        """
        if self.providers:
            return self.providers[0]
        return None


def _install_fake_providers(monkeypatch: pytest.MonkeyPatch, providers: list[_FakeProvider]) -> None:
    """Replace the provider registry with a deterministic fake.

    Args:
        monkeypatch: Active pytest monkeypatch fixture.
        providers: Providers the fake registry exposes.
    """
    monkeypatch.setattr(
        "chibi.services.providers.RegisteredProviders",
        functools.partial(_FakeProviderRegistry, providers=providers),
    )


class TestUserLlmDefaultFallback:
    """Test LLM provider and model resolution after the user-level fields were removed."""

    def test_new_thread_resolves_provider_from_settings_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a thread without a selection resolves the configured default provider."""
        alpha = _FakeProvider(name="alpha")
        beta = _FakeProvider(name="beta")
        _install_fake_providers(monkeypatch, [beta, alpha])
        monkeypatch.setattr(gpt_settings, "default_provider", "alpha")
        user = User(id=1)

        assert user.get_active_llm_provider(thread_id=42) is alpha

    def test_provider_falls_back_to_first_chat_ready_without_settings_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that provider resolution ends at the first chat-ready provider."""
        beta = _FakeProvider(name="beta")
        _install_fake_providers(monkeypatch, [beta])
        monkeypatch.setattr(gpt_settings, "default_provider", None)
        user = User(id=1)

        assert user.get_active_llm_provider(thread_id=42) is beta

    def test_no_provider_raises_when_nothing_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that an empty registry and an unset default leave no provider to resolve."""
        _install_fake_providers(monkeypatch, [])
        monkeypatch.setattr(gpt_settings, "default_provider", None)
        user = User(id=1)

        with pytest.raises(NoProviderSelectedError):
            user.get_active_llm_provider(thread_id=42)

    def test_model_falls_back_to_settings_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a thread without a selection resolves the configured default model."""
        _install_fake_providers(monkeypatch, [_FakeProvider(name="alpha")])
        monkeypatch.setattr(gpt_settings, "default_model", "alpha-model")
        user = User(id=1)

        assert user.get_active_llm_model(thread_id=42) == "alpha-model"

    def test_model_is_none_without_settings_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that no selection and no configured default leaves the model unset."""
        _install_fake_providers(monkeypatch, [_FakeProvider(name="alpha")])
        monkeypatch.setattr(gpt_settings, "default_model", None)
        user = User(id=1)

        assert user.get_active_llm_model(thread_id=42) is None

    def test_unset_model_lets_call_site_use_provider_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the unset-model contract leaves the provider default to the call site."""
        provider = _FakeProvider(name="alpha", default_model="alpha-builtin-model")
        _install_fake_providers(monkeypatch, [provider])
        monkeypatch.setattr(gpt_settings, "default_provider", "alpha")
        monkeypatch.setattr(gpt_settings, "default_model", None)
        user = User(id=1)
        active_provider = user.get_active_llm_provider(thread_id=42)

        assert (user.get_active_llm_model(thread_id=42) or active_provider.default_model) == "alpha-builtin-model"

    def test_legacy_record_with_removed_fields_ignores_them(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a stored record carrying the removed fields loads and its stale values are ignored."""
        stale = _FakeProvider(name="stale-provider")
        alpha = _FakeProvider(name="alpha")
        _install_fake_providers(monkeypatch, [stale, alpha])
        monkeypatch.setattr(gpt_settings, "default_provider", "alpha")
        monkeypatch.setattr(gpt_settings, "default_model", "alpha-model")
        legacy_record = {
            "id": 7,
            "gpt_model": "stale-model",
            "selected_gpt_model_name": "stale-model",
            "selected_gpt_provider_name": "stale-provider",
        }

        user = User.model_validate_json(json.dumps(legacy_record))

        assert not hasattr(user, "gpt_model")
        assert not hasattr(user, "selected_gpt_model_name")
        assert not hasattr(user, "selected_gpt_provider_name")
        assert user.thread_selected_llm == {}
        assert user.get_active_llm_provider(thread_id=42) is alpha
        assert user.get_active_llm_model(thread_id=42) == "alpha-model"

    def test_thread_selection_wins_over_settings_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a thread-level selection beats both configured defaults."""
        alpha = _FakeProvider(name="alpha")
        beta = _FakeProvider(name="beta")
        _install_fake_providers(monkeypatch, [alpha, beta])
        monkeypatch.setattr(gpt_settings, "default_provider", "alpha")
        monkeypatch.setattr(gpt_settings, "default_model", "alpha-model")
        user = User(id=1)
        user.thread_selected_llm[42] = SelectedModel(name="thread-model", provider_name="beta")

        assert user.get_active_llm_provider(thread_id=42) is beta
        assert user.get_active_llm_model(thread_id=42) == "thread-model"
