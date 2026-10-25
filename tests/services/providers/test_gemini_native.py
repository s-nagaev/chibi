"""Unit tests for the Gemini provider: OCR, overflow handling and capability gating."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.errors import APIError
from google.genai.types import (
    Candidate,
    Content,
    ContentDict,
    ContentListUnionDict,
    GenerateContentConfig,
    GenerateContentResponse,
    Part,
)

from chibi.exceptions import ContextLengthExceededError, ServiceResponseError
from chibi.models import Message, User
from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.gemini_native import Gemini, get_google_model_capabilities

# Sample PDF bytes for testing
SAMPLE_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"


# Use a test token for provider instantiation
TEST_TOKEN = "test-token-for-ocr"


@pytest.mark.asyncio
async def test_ocr_basic_pdf_extraction():
    """Test basic PDF text extraction with ocr method."""
    provider = Gemini(token=TEST_TOKEN)

    # Create mock response
    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "A test PDF", '
        '"full_description": "This is a test PDF document", '
        '"text": "Extracted text from PDF"}'
    )

    # Patch _generate_content to return our mock response
    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify the result
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "A test PDF"
    assert result.full_description == "This is a test PDF document"
    assert result.text == "Extracted text from PDF"


@pytest.mark.asyncio
async def test_ocr_with_custom_model():
    """Test ocr method with custom model parameter."""
    provider = Gemini(token=TEST_TOKEN)

    captured_model = None

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Custom model test", '
        '"full_description": "Testing with custom model", '
        '"text": "Custom model extracted text"}'
    )

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_model
        captured_model = model
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        result = await provider.ocr(
            pdf=SAMPLE_PDF_BYTES,
            model="custom-ocr-model",
        )

    assert captured_model == "custom-ocr-model"
    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_ocr_default_model():
    """Test that default vision model is used when none specified."""
    provider = Gemini(token=TEST_TOKEN)

    captured_model = None

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Test PDF", "full_description": "Test PDF description", "text": "Test text"}'
    )

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_model
        captured_model = model
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert captured_model == provider.default_vision_model


@pytest.mark.asyncio
async def test_ocr_pdf_bytes_passed_correctly():
    """Test that PDF bytes are correctly passed via Part.from_bytes."""
    provider = Gemini(token=TEST_TOKEN)

    captured_contents = None
    captured_mime_type = None

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_contents, captured_mime_type
        captured_contents = contents
        # Check the Part.from_bytes call in contents
        if isinstance(contents, list) and len(contents) > 0:
            part = contents[0]
            if hasattr(part, "mime_type"):
                captured_mime_type = part.mime_type
            # Also capture the data if available
            if hasattr(part, "data"):
                captured_mime_type = part.mime_type
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        test_pdf = b"%PDF-1.4 test pdf content"
        await provider.ocr(pdf=test_pdf)

    # Verify that contents is a list with the PDF part
    assert isinstance(captured_contents, list)
    assert len(captured_contents) >= 1


@pytest.mark.asyncio
async def test_ocr_part_from_bytes_mime_type():
    """Test that Part.from_bytes is called with correct mime_type='application/pdf'."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    # We need to verify Part.from_bytes is called with the right mime_type
    # The ocr method builds contents as: [Part.from_bytes(data=pdf, mime_type="application/pdf"), prompt]
    with patch("chibi.services.providers.gemini_native.Part.from_bytes") as mock_from_bytes:
        mock_part = MagicMock()
        mock_part.mime_type = "application/pdf"
        mock_from_bytes.return_value = mock_part

        with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response
            await provider.ocr(pdf=SAMPLE_PDF_BYTES)

        # Verify Part.from_bytes was called with correct mime_type
        mock_from_bytes.assert_called_once()
        call_kwargs = mock_from_bytes.call_args.kwargs
        assert call_kwargs.get("mime_type") == "application/pdf"
        assert call_kwargs.get("data") == SAMPLE_PDF_BYTES


@pytest.mark.asyncio
async def test_ocr_empty_response_raises_error():
    """Test that empty response raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = None

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(ServiceResponseError) as exc_info:
            await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "empty" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_parse_error_raises_error():
    """Test that parse error raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = "invalid json that cannot be parsed"

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(ServiceResponseError) as exc_info:
            await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "parse" in str(exc_info.value).lower() or "extract" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_returns_vision_result_schema():
    """Test that ocr method returns VisionResultSchema instance."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Invoice", '
        '"full_description": "Invoice for services", '
        '"text": "Invoice #12345\\nDate: 2024-01-01\\nAmount: $100.00"}'
    )

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "Invoice"
    assert result.full_description == "Invoice for services"
    assert result.text is not None and "Invoice #12345" in result.text


@pytest.mark.asyncio
async def test_ocr_with_text_extraction():
    """Test ocr method when PDF contains text."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Document", '
        '"full_description": "A legal document", '
        '"text": "Important legal text extracted from the PDF document"}'
    )

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert isinstance(result, VisionResultSchema)
    assert result.text == "Important legal text extracted from the PDF document"


@pytest.mark.asyncio
async def test_ocr_optional_text_field():
    """Test that ocr works when text field is null/optional."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Image scan", "full_description": "Scanned image without text", "text": null}'
    )

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert isinstance(result, VisionResultSchema)
    assert result.text is None


@pytest.mark.asyncio
async def test_ocr_mock_generate_content_called():
    """Test that _generate_content is called correctly with expected parameters."""
    provider = Gemini(token=TEST_TOKEN)

    call_args = {}

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    async def mock_generate_content(self, model, contents, config):
        call_args["model"] = model
        call_args["contents"] = contents
        call_args["config"] = config
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify _generate_content was called
    assert "model" in call_args
    assert "contents" in call_args
    assert "config" in call_args

    # Verify model is set
    assert call_args["model"] == provider.default_vision_model

    # Verify contents structure - should be a list with Part and prompt
    assert isinstance(call_args["contents"], list)
    assert len(call_args["contents"]) == 2  # Part.from_bytes + prompt string


@pytest.mark.asyncio
async def test_ocr_config_has_response_schema():
    """Test that generation config includes VisionResultSchema as response_schema."""
    provider = Gemini(token=TEST_TOKEN)

    captured_config = None

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_config
        captured_config = config
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify config has response_schema set to VisionResultSchema
    assert captured_config is not None
    assert hasattr(captured_config, "response_schema")
    assert captured_config.response_schema == VisionResultSchema


def _make_gemini_api_error(status_code: int, message: str) -> APIError:
    """Build a google-genai APIError with the given status code and message."""
    return APIError(
        code=status_code,
        response_json={
            "error": {
                "code": status_code,
                "message": message,
                "status": "INVALID_ARGUMENT",
            }
        },
    )


class _FakeAio:
    """Async context manager yielding a mocked Gemini client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __aenter__(self) -> Any:
        return self._client

    async def __aexit__(self, *args: Any) -> bool:
        return False


@pytest.mark.asyncio
async def test_overflow_400_raises_context_length_exceeded_error():
    """A Gemini 400 whose body reports a token-count overflow raises ContextLengthExceededError."""
    provider = Gemini(token=TEST_TOKEN)
    overflow_error = _make_gemini_api_error(
        400,
        "The input token count (123456) exceeds the maximum number of tokens allowed (100000).",
    )
    mock_client = MagicMock()
    mock_client.models.generate_content = AsyncMock(side_effect=overflow_error)
    fake_client_cls = MagicMock(return_value=SimpleNamespace(aio=_FakeAio(mock_client)))

    with (
        patch("chibi.services.providers.gemini_native.Client", fake_client_cls),
        pytest.raises(ContextLengthExceededError),
    ):
        await provider._generate_content(
            model="models/gemini-3.8-flash",
            contents=cast(ContentListUnionDict, [{"role": "user", "parts": [{"text": "Hi"}]}]),
            config=GenerateContentConfig(),
            retry_server_errors=True,
        )

    assert mock_client.models.generate_content.await_count == 1


@pytest.mark.asyncio
async def test_non_overflow_400_still_raises_service_response_error():
    """A Gemini 400 unrelated to context length keeps the generic conversion."""
    provider = Gemini(token=TEST_TOKEN)
    invalid_error = _make_gemini_api_error(400, "Invalid value at 'contents[0].parts[0]'")
    mock_client = MagicMock()
    mock_client.models.generate_content = AsyncMock(side_effect=invalid_error)
    fake_client_cls = MagicMock(return_value=SimpleNamespace(aio=_FakeAio(mock_client)))

    with (
        patch("chibi.services.providers.gemini_native.Client", fake_client_cls),
        pytest.raises(ServiceResponseError),
    ):
        await provider._generate_content(
            model="models/gemini-3.8-flash",
            contents=cast(ContentListUnionDict, [{"role": "user", "parts": [{"text": "Hi"}]}]),
            config=GenerateContentConfig(),
            retry_server_errors=True,
        )


GATE_SYSTEM_PROMPT = "You are a capability-gate system prompt."

GEMINI_MODEL = "models/gemini-3.8-flash"
GEMMA4_MODEL = "models/gemma-4-31b-it"
GEMMA3_MODEL = "models/gemma-3-27b-it"
UNKNOWN_GOOGLE_MODEL = "models/gemma-5-preview"


def _make_text_response(answer: str) -> GenerateContentResponse:
    """Build a GenerateContentResponse carrying a plain text answer.

    Args:
        answer: The model answer text.

    Returns:
        A single-candidate response with one text part.
    """
    return GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=[Part(text=answer)]))])


class _ChatCapture:
    """Container for the config and contents captured from the mocked completion call."""

    def __init__(self) -> None:
        self.config: GenerateContentConfig | None = None
        self.contents: list[ContentDict] | None = None


async def _run_gemini_chat(model: str, messages: list[Message]) -> _ChatCapture:
    """Drive ``get_chat_response`` with a mocked ``_generate_content`` and capture the payload.

    Args:
        model: The chat model identifier to request.
        messages: The dialogue history to send.

    Returns:
        The captured generation config and converted contents.
    """
    capture = _ChatCapture()

    async def mock_generate_content(
        self: Any, model: str, contents: Any, config: GenerateContentConfig, **kwargs: Any
    ) -> GenerateContentResponse:
        capture.config = config
        capture.contents = contents
        return _make_text_response("ok")

    provider = Gemini(token=TEST_TOKEN)
    with (
        patch(
            "chibi.services.providers.gemini_native.prepare_system_prompt",
            new=AsyncMock(return_value=GATE_SYSTEM_PROMPT),
        ),
        patch.object(Gemini, "_generate_content", mock_generate_content),
    ):
        await provider.get_chat_response(
            messages=messages,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=model,
            system_prompt=GATE_SYSTEM_PROMPT,
        )
    return capture


@pytest.mark.asyncio
async def test_gemini_model_keeps_system_instruction_and_tools():
    """A Gemini model keeps the system instruction and the tools, with no payload change."""
    capture = await _run_gemini_chat(GEMINI_MODEL, [Message(role="user", content="Hello")])

    assert capture.config is not None
    assert capture.config.system_instruction == GATE_SYSTEM_PROMPT
    assert capture.config.tools is not None
    assert capture.contents is not None
    assert capture.contents[0]["parts"] == [{"text": "Hello"}]


@pytest.mark.asyncio
async def test_gemma4_uses_system_channel_with_tools():
    """Gemma 4 is treated like Gemini: system channel with tools."""
    capture = await _run_gemini_chat(GEMMA4_MODEL, [Message(role="user", content="Hello")])

    assert capture.config is not None
    assert capture.config.system_instruction == GATE_SYSTEM_PROMPT
    assert capture.config.tools is not None
    assert capture.contents is not None
    assert capture.contents[0]["parts"] == [{"text": "Hello"}]


@pytest.mark.asyncio
async def test_gemma3_gets_no_system_instruction_and_no_tools():
    """Gemma 3 receives neither a system instruction nor tools."""
    capture = await _run_gemini_chat(GEMMA3_MODEL, [Message(role="user", content="Hello")])

    assert capture.config is not None
    assert capture.config.system_instruction is None
    assert capture.config.tools is None


@pytest.mark.asyncio
async def test_gemma3_prompt_folds_into_first_user_message_only():
    """Gemma 3 receives the prompt as a leading text part of the first user message only."""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
        Message(role="user", content="Second question"),
    ]
    capture = await _run_gemini_chat(GEMMA3_MODEL, messages)

    assert capture.contents is not None
    assert capture.contents[0]["role"] == "user"
    first_parts = capture.contents[0]["parts"]
    assert first_parts is not None
    assert first_parts[0] == {"text": GATE_SYSTEM_PROMPT}
    assert first_parts[1] == {"text": "Hello"}
    assert capture.contents[2]["parts"] == [{"text": "Second question"}]


@pytest.mark.asyncio
async def test_gemma3_folded_prompt_is_not_persisted():
    """The folded prompt lives only in the request payload, never in the returned history."""
    provider = Gemini(token=TEST_TOKEN)

    async def mock_generate_content(
        self: Any, model: str, contents: Any, config: GenerateContentConfig, **kwargs: Any
    ) -> GenerateContentResponse:
        return _make_text_response("ok")

    with (
        patch(
            "chibi.services.providers.gemini_native.prepare_system_prompt",
            new=AsyncMock(return_value=GATE_SYSTEM_PROMPT),
        ),
        patch.object(Gemini, "_generate_content", mock_generate_content),
    ):
        _chat_response, new_messages = await provider.get_chat_response(
            messages=[Message(role="user", content="Hello")],
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=GEMMA3_MODEL,
            system_prompt=GATE_SYSTEM_PROMPT,
        )

    assert [(msg.role, msg.content) for msg in new_messages] == [("assistant", "ok")]
    assert all(GATE_SYSTEM_PROMPT not in (msg.content or "") for msg in new_messages)


@pytest.mark.asyncio
async def test_unknown_google_model_defaults_to_full_capability():
    """A model matching no capability row gets the system channel and tools, like Gemini."""
    capture = await _run_gemini_chat(UNKNOWN_GOOGLE_MODEL, [Message(role="user", content="Hello")])

    assert capture.config is not None
    assert capture.config.system_instruction == GATE_SYSTEM_PROMPT
    assert capture.config.tools is not None
    assert capture.contents is not None
    assert capture.contents[0]["parts"] == [{"text": "Hello"}]


@pytest.mark.asyncio
async def test_gemma3_without_user_message_does_not_crash():
    """A payload without a usable user message skips the fold instead of crashing."""
    messages = [Message(role="assistant", content="only an assistant turn")]
    capture = await _run_gemini_chat(GEMMA3_MODEL, messages)

    assert capture.config is not None
    assert capture.config.system_instruction is None
    assert capture.config.tools is None


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("models/gemini-3.8-flash", (True, True)),
        ("gemini-2.5-flash", (True, True)),
        ("models/gemma-4-31b-it", (True, True)),
        ("models/gemma-3-27b-it", (False, False)),
        ("models/gemma-3n-e4b-it", (False, False)),
        ("models/gemma-2-2b-it", (False, False)),
        ("models/gemma-1b", (False, False)),
        ("models/gemma-31", (True, True)),
        ("models/gemma-5-preview", (True, True)),
        ("models/some-unknown-model", (True, True)),
        (None, (True, True)),
    ],
)
def test_get_google_model_capabilities_resolution(model_name: str | None, expected: tuple[bool, bool]) -> None:
    """The capability table resolves families by longest boundary-respecting prefix."""
    assert get_google_model_capabilities(model_name) == expected
