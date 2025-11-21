"""Tests for chibi.models module."""

import json

from google.genai.types import ContentDict, PartDict

from chibi.models import FunctionSchema, Message, ToolSchema


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
                    "function_call": {"name": "calculate", "args": {"expression": "2 + 2"}},
                    "thought_signature": None,
                }
            ],
        }
        assert google_content == expected

    def test_to_google_tool_response(self):
        """Test converting tool response message to Google format."""
        message = Message(
            role="tool", content="The weather in San Francisco is sunny, 72°F", tool_call_id="get_weather"
        )
        google_content = message.to_google()

        expected = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_weather",
                        "response": {"content": "The weather in San Francisco is sunny, 72°F"},
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
        google_content = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_weather",
                        "response": {"content": "The weather in San Francisco is sunny, 72°F"},
                    }
                }
            ],
        }
        message = Message.from_google(google_content)

        assert message.role == "tool"
        assert message.content == "The weather in San Francisco is sunny, 72°F"
        assert message.tool_call_id == "get_weather"
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
        original = Message(role="tool", content="Weather is sunny", tool_call_id="get_weather")
        google_content = original.to_google()
        converted = Message.from_google(google_content)

        assert converted.role == original.role
        assert converted.content == original.content
        assert converted.tool_call_id == original.tool_call_id
        assert converted.tool_calls is None

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
                    "function_call": {"name": "ping", "args": {}},
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
