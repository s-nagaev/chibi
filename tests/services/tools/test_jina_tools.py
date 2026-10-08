"""Tests for the Jina web tools (jina_read, jina_search)."""

import importlib
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from chibi.config import gpt_settings
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.jina import (
    JINA_SEARCH_RESULT_TRUNCATION_LIMIT,
    JinaReadTool,
    JinaSearchTool,
    _parse_search_results,
    _truncate_result_content,
)
from chibi.services.providers.tools.tool import RegisteredChibiTools


def _mock_transport(handler: Any) -> Any:
    """Return a factory usable in place of httpx.AsyncHTTPTransport."""
    return lambda retries, proxy: httpx.MockTransport(handler)


class TestJinaReadValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(ToolException, match="absolute http\\(s\\) URL"):
            await JinaReadTool.function(url="ftp://example.com/file")

    @pytest.mark.asyncio
    async def test_rejects_url_without_netloc(self) -> None:
        with pytest.raises(ToolException, match="absolute http\\(s\\) URL"):
            await JinaReadTool.function(url="not-a-url")

    @pytest.mark.asyncio
    async def test_rejects_unsupported_return_format(self) -> None:
        with pytest.raises(ToolException, match="Unsupported return_format"):
            await JinaReadTool.function(url="https://example.com", return_format="pdf")


class TestJinaReadHeadersAndResult:
    @pytest.mark.asyncio
    async def test_header_mapping(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(dict(request.headers))
            return httpx.Response(200, text="Title: Example\n\nMarkdown Content: hello")

        with (
            patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)),
            patch.object(gpt_settings, "jina_api_key", "test-key"),
        ):
            result = await JinaReadTool.function(
                url="https://example.com/page",
                return_format="markdown",
                target_selector="article",
                with_links_summary=True,
                timeout=30,
            )

        assert result["content"].startswith("Title: Example")
        assert captured["x-return-format"] == "markdown"
        assert captured["x-target-selector"] == "article"
        assert captured["x-with-links-summary"] == "true"
        assert captured["authorization"] == "Bearer test-key"
        assert captured["x-return-format"] == "markdown"
        assert captured.get("accept-encoding", "gzip, deflate") == "gzip, deflate"

    @pytest.mark.asyncio
    async def test_no_authorization_without_key(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(dict(request.headers))
            return httpx.Response(200, text="content")

        with (
            patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)),
            patch.object(gpt_settings, "jina_api_key", None),
        ):
            await JinaReadTool.function(url="https://example.com")

        assert "authorization" not in captured
        assert captured["x-return-format"] == "text"

    @pytest.mark.asyncio
    async def test_empty_response_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="")

        with patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)):
            with pytest.raises(ToolException, match="empty response"):
                await JinaReadTool.function(url="https://example.com")


class TestJinaSearch:
    @pytest.mark.asyncio
    async def test_requires_key(self) -> None:
        with (
            patch.object(gpt_settings, "jina_api_key", None),
            pytest.raises(ToolException, match="JINA_API_KEY"),
        ):
            await JinaSearchTool.function(search_phrase="test")

    @pytest.mark.asyncio
    async def test_successful_search_truncates_content(self) -> None:
        captured: dict[str, Any] = {}
        long_content = "x" * (JINA_SEARCH_RESULT_TRUNCATION_LIMIT + 500)
        payload = [{"title": "Result", "url": "https://example.com", "content": long_content}]

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = dict(request.headers).get("authorization")
            return httpx.Response(200, json=payload)

        with (
            patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)),
            patch.object(gpt_settings, "jina_api_key", "test-key"),
        ):
            result = await JinaSearchTool.function(search_phrase="chibi bot")

        assert captured["url"] == "https://s.jina.ai"
        assert captured["auth"] == "Bearer test-key"
        results = result["search_results"]
        assert len(results) == 1
        assert results[0]["title"] == "Result"
        assert results[0]["url"] == "https://example.com"
        assert len(results[0]["content"]) <= JINA_SEARCH_RESULT_TRUNCATION_LIMIT + len("... (truncated)")

    @pytest.mark.asyncio
    async def test_empty_results_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[])

        with (
            patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)),
            patch.object(gpt_settings, "jina_api_key", "test-key"),
        ):
            result = await JinaSearchTool.function(search_phrase="nothing here")

        assert result["search_results"] == "Ooops, the search returned an empty list of results."


class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_401_suggests_key_and_fallback(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        with (
            patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)),
            patch.object(gpt_settings, "jina_api_key", "stale-key"),
        ):
            with pytest.raises(ToolException, match="ddgs_web_search"):
                await JinaSearchTool.function(search_phrase="q")

    @pytest.mark.asyncio
    async def test_429_reports_rate_limits(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")

        with patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)):
            with pytest.raises(ToolException, match="Too Many Requests"):
                await JinaReadTool.function(url="https://example.com")

    @pytest.mark.asyncio
    async def test_timeout_is_mapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("too slow", request=request)

        with patch("httpx.AsyncHTTPTransport", new=_mock_transport(handler)):
            with pytest.raises(ToolException, match="timed out"):
                await JinaReadTool.function(url="https://example.com")


class TestParsingHelpers:
    def test_truncate_short_text_untouched(self) -> None:
        assert _truncate_result_content(text="short") == "short"

    def test_truncate_long_text(self) -> None:
        truncated = _truncate_result_content(text="y" * 3000)
        assert truncated.startswith("y" * JINA_SEARCH_RESULT_TRUNCATION_LIMIT)
        assert truncated.endswith("(truncated)")

    def test_parse_json_array(self) -> None:
        raw = '[{"title": "T", "url": "https://a", "content": "c"}, "garbage"]'
        results = _parse_search_results(raw=raw)
        assert results == [{"title": "T", "url": "https://a", "content": "c"}]

    def test_parse_data_object(self) -> None:
        raw = '{"data": [{"title": "T", "url": "https://a", "content": "c"}]}'
        assert len(_parse_search_results(raw=raw)) == 1

    def test_parse_non_json_falls_back_to_single_result(self) -> None:
        results = _parse_search_results(raw="plain markdown text")
        assert len(results) == 1
        assert results[0]["content"] == "plain markdown text"

    def test_parse_empty_body(self) -> None:
        assert _parse_search_results(raw="   ") == []


class TestKeyGatedRegistration:
    def test_jina_search_registers_only_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import chibi.services.providers.tools.jina as jina_module

        try:
            monkeypatch.setattr(gpt_settings, "jina_api_key", "test-key")
            reloaded = importlib.reload(jina_module)
            assert reloaded.JinaSearchTool.register is True
            assert reloaded.JinaReadTool.register is True

            RegisteredChibiTools.tools_map.pop("jina_search", None)
            monkeypatch.setattr(gpt_settings, "jina_api_key", None)
            reloaded = importlib.reload(jina_module)
            assert reloaded.JinaSearchTool.register is False
            assert reloaded.JinaReadTool.register is True
        finally:
            monkeypatch.undo()
            RegisteredChibiTools.tools_map.pop("jina_read", None)
            RegisteredChibiTools.tools_map.pop("jina_search", None)
            importlib.reload(jina_module)
