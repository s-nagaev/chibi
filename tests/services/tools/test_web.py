"""Tests for the hybrid read_web_page pipeline: quality gate, circuit breaker, routing."""

from typing import Any
from unittest.mock import patch

import pytest

from chibi.config import gpt_settings
from chibi.services.providers.tools import web
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.web import (
    _CircuitBreaker,
    _looks_like_html,
    _looks_like_js_garbage,
    _passes_quality_gate,
    _strip_jina_preamble,
)

LONG_TEXT = "This is a readable article about cycling endurance training. " * 10
RAW_HTML = f"<!DOCTYPE html><html><body><p>{'x' * 300}</p></body></html>"


@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset module-level breakers around every test."""
    web._trafilatura_breaker.reset()
    web._jina_breaker.reset()
    yield
    web._trafilatura_breaker.reset()
    web._jina_breaker.reset()


class TestQualityGate:
    def test_none_fails(self) -> None:
        assert _passes_quality_gate(text=None) is False

    def test_short_text_fails(self) -> None:
        assert _passes_quality_gate(text="too short") is False

    def test_html_fallback_fails(self) -> None:
        assert _passes_quality_gate(text=RAW_HTML) is False

    def test_text_identical_to_raw_html_fails(self) -> None:
        assert _passes_quality_gate(text=LONG_TEXT.strip(), raw_html=LONG_TEXT.strip()) is False

    def test_js_garbage_fails(self) -> None:
        garbage = "function() { document.getElementById('a').innerHTML = 'x'; } " * 30
        assert _passes_quality_gate(text=garbage) is False

    def test_coherent_text_passes(self) -> None:
        assert _passes_quality_gate(text=LONG_TEXT, raw_html=RAW_HTML) is True

    def test_empty_string_fails(self) -> None:
        assert _passes_quality_gate(text="") is False


class TestHelpers:
    def test_looks_like_html_positive(self) -> None:
        assert _looks_like_html(text=RAW_HTML) is True
        assert _looks_like_html(text="<html lang='en'><body>hi</body></html>") is True

    def test_looks_like_html_negative(self) -> None:
        assert _looks_like_html(text=LONG_TEXT) is False
        assert _looks_like_html(text="<b>just a bold tag in prose</b> and more prose " * 20) is False

    def test_looks_like_js_garbage_positive(self) -> None:
        assert _looks_like_js_garbage(text="window.addEventListener('load', () => {}) " * 30) is True

    def test_looks_like_js_garbage_negative(self) -> None:
        assert _looks_like_js_garbage(text=LONG_TEXT) is False

    def test_strip_jina_preamble(self) -> None:
        raw = "Title: Example\nURL Source: https://example.com\nMarkdown Content:\n\nBody of the page"
        assert _strip_jina_preamble(text=raw) == "Body of the page"

    def test_strip_jina_preamble_without_marker(self) -> None:
        assert _strip_jina_preamble(text="plain text") == "plain text"


class FakeClock:
    """Injectable monotonic clock seam for breaker tests."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCircuitBreaker:
    def test_opens_after_threshold_failures(self) -> None:
        breaker = _CircuitBreaker()
        assert breaker.allow_request() is True
        for _ in range(web.BREAKER_FAILURE_THRESHOLD):
            breaker.record_failure()
        assert breaker.allow_request() is False

    def test_stays_closed_below_threshold(self) -> None:
        breaker = _CircuitBreaker()
        for _ in range(web.BREAKER_FAILURE_THRESHOLD - 1):
            breaker.record_failure()
        assert breaker.allow_request() is True

    def test_blocks_during_cooldown_then_half_open_trial(self) -> None:
        clock = FakeClock()
        breaker = _CircuitBreaker()
        with patch.object(web, "_clock", clock):
            for _ in range(web.BREAKER_FAILURE_THRESHOLD):
                breaker.record_failure()
            assert breaker.allow_request() is False

            clock.advance(web.BREAKER_COOLDOWN_SECONDS - 1)
            assert breaker.allow_request() is False

            clock.advance(1)
            assert breaker.allow_request() is True
            assert breaker.allow_request() is False

    def test_successful_trial_closes_circuit(self) -> None:
        clock = FakeClock()
        breaker = _CircuitBreaker()
        with patch.object(web, "_clock", clock):
            for _ in range(web.BREAKER_FAILURE_THRESHOLD):
                breaker.record_failure()
            clock.advance(web.BREAKER_COOLDOWN_SECONDS)
            assert breaker.allow_request() is True
            breaker.record_success()
            assert breaker.allow_request() is True
            breaker.record_failure()
            assert breaker.allow_request() is True

    def test_failed_trial_reopens_circuit(self) -> None:
        clock = FakeClock()
        breaker = _CircuitBreaker()
        with patch.object(web, "_clock", clock):
            for _ in range(web.BREAKER_FAILURE_THRESHOLD):
                breaker.record_failure()
            clock.advance(web.BREAKER_COOLDOWN_SECONDS)
            assert breaker.allow_request() is True
            breaker.record_failure()
            assert breaker.allow_request() is False

            clock.advance(web.BREAKER_COOLDOWN_SECONDS)
            assert breaker.allow_request() is True


class TestHybridRouting:
    @pytest.mark.asyncio
    async def test_without_key_trafilatura_first(self) -> None:
        jina = _async_return(("jina text " * 50, None))
        trafilatura = _async_return((LONG_TEXT, RAW_HTML))
        with (
            patch.object(gpt_settings, "jina_api_key", None),
            patch.object(web, "_fetch_via_trafilatura", trafilatura),
            patch.object(web, "_fetch_via_jina", jina),
        ):
            result = await web.ReadWebPageTool.function(url="https://example.com")

        assert result["content"] == LONG_TEXT
        assert jina.await_count == 0
        assert trafilatura.await_count == 1

    @pytest.mark.asyncio
    async def test_with_key_jina_first(self) -> None:
        jina = _async_return((LONG_TEXT, None))
        trafilatura = _async_return((LONG_TEXT, RAW_HTML))
        with (
            patch.object(gpt_settings, "jina_api_key", "key"),
            patch.object(web, "_fetch_via_trafilatura", trafilatura),
            patch.object(web, "_fetch_via_jina", jina),
        ):
            result = await web.ReadWebPageTool.function(url="https://example.com")

        assert result["content"] == LONG_TEXT
        assert jina.await_count == 1
        assert trafilatura.await_count == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_secondary_on_primary_gate_failure(self) -> None:
        trafilatura = _async_return(("t" * 50, "t" * 50))
        jina = _async_return((LONG_TEXT, None))
        with (
            patch.object(gpt_settings, "jina_api_key", None),
            patch.object(web, "_fetch_via_trafilatura", trafilatura),
            patch.object(web, "_fetch_via_jina", jina),
        ):
            result = await web.ReadWebPageTool.function(url="https://example.com")

        assert result["content"] == LONG_TEXT
        assert jina.await_count == 1
        assert trafilatura.await_count == 1

    @pytest.mark.asyncio
    async def test_falls_back_on_primary_exception(self) -> None:
        async def failing_jina(url: str) -> tuple[str, Any]:
            raise ToolException("jina down")

        trafilatura = _async_return((LONG_TEXT, RAW_HTML))
        with (
            patch.object(gpt_settings, "jina_api_key", "key"),
            patch.object(web, "_fetch_via_trafilatura", trafilatura),
            patch.object(web, "_fetch_via_jina", failing_jina),
        ):
            result = await web.ReadWebPageTool.function(url="https://example.com")

        assert result["content"] == LONG_TEXT

    @pytest.mark.asyncio
    async def test_open_circuit_skips_path(self) -> None:
        async def failing_jina(url: str) -> tuple[str, Any]:
            raise ToolException("jina down")

        jina = AsyncFetchStub(failing_jina)
        trafilatura = _async_return((LONG_TEXT, RAW_HTML))
        with (
            patch.object(gpt_settings, "jina_api_key", "key"),
            patch.object(web, "_fetch_via_trafilatura", trafilatura),
            patch.object(web, "_fetch_via_jina", jina),
        ):
            for _ in range(web.BREAKER_FAILURE_THRESHOLD):
                await web.ReadWebPageTool.function(url="https://example.com")

            result = await web.ReadWebPageTool.function(url="https://example.com")

        assert result["content"] == LONG_TEXT
        assert jina.await_count == web.BREAKER_FAILURE_THRESHOLD
        assert web._jina_breaker.allow_request() is False

    @pytest.mark.asyncio
    async def test_both_paths_fail_gate_returns_best_raw(self) -> None:
        jina = _async_return(("j" * 50, None))
        trafilatura = _async_return(("t" * 500, "t" * 500))
        with (
            patch.object(gpt_settings, "jina_api_key", "key"),
            patch.object(web, "_fetch_via_trafilatura", trafilatura),
            patch.object(web, "_fetch_via_jina", jina),
        ):
            result = await web.ReadWebPageTool.function(url="https://example.com")

        assert result["data"] == "t" * 500
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_all_paths_exhausted_raises(self) -> None:
        async def failing_jina(url: str) -> tuple[str, Any]:
            raise ToolException("jina down")

        async def failing_trafilatura(url: str) -> tuple[str, Any]:
            raise ToolException("trafilatura down")

        with (
            patch.object(gpt_settings, "jina_api_key", "key"),
            patch.object(web, "_fetch_via_trafilatura", failing_trafilatura),
            patch.object(web, "_fetch_via_jina", failing_jina),
        ):
            with pytest.raises(ToolException, match="All available extraction paths failed"):
                await web.ReadWebPageTool.function(url="https://example.com")


def _async_return(value: tuple[str, Any]) -> Any:
    """Build an AsyncMock returning a fixed (text, raw_html) tuple."""

    async def fetcher(url: str) -> tuple[str, Any]:
        return value

    return AsyncFetchStub(fetcher)


class AsyncFetchStub:
    def __init__(self, fetcher: Any) -> None:
        self._fetcher = fetcher
        self.await_count = 0

    async def __call__(self, url: str) -> tuple[str, Any]:
        self.await_count += 1
        return await self._fetcher(url)
