"""Unit tests for the Gemini JSON-Schema normalization and per-declaration isolation.

Covers the oneOf crash fix: google-genai's strict ``Schema`` model has no
``oneOf`` field (``extra='forbid'``), so MCP tool schemas carrying ``oneOf``
(the ``mcp_melious_rest_scrape_url`` shape) used to crash the whole Gemini
``tools_list`` conversion and take the provider down.
"""

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.genai.types import FunctionDeclaration

from chibi.services.providers.gemini_native import Gemini, _normalize_schema_for_gemini
from chibi.services.providers.tools import RegisteredChibiTools

TEST_TOKEN = "test-token-for-gemini-schema"

SCRAPE_URL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "format": "uri", "description": "URL to fetch."},
        "urls": {
            "type": "array",
            "items": {"type": "string", "format": "uri"},
            "maxItems": 10,
            "description": "Multiple URLs to fetch in one request.",
        },
        "query": {
            "description": "Focus extraction on content relevant to this query.",
            "type": "string",
        },
    },
    "required": ["url"],
    "additionalProperties": False,
    "oneOf": [
        {"required": ["url"]},
        {"required": ["urls"]},
    ],
}


class TestNormalizeSchemaForGemini:
    """Unit tests for the recursive schema normalizer."""

    def test_oneof_mapped_to_anyof(self) -> None:
        schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}

        normalized = _normalize_schema_for_gemini(schema)

        assert normalized == {"anyOf": [{"type": "string"}, {"type": "number"}]}
        assert "oneOf" not in normalized

    def test_recursion_into_properties_items_and_branches(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "one_of_field": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                "list_field": {"type": "array", "items": {"oneOf": [{"type": "string"}]}},
            },
            "anyOf": [{"properties": {"nested": {"oneOf": [{"type": "integer"}]}}}],
            "$defs": {"inner": {"oneOf": [{"type": "boolean"}]}},
            "definitions": {"inner2": {"oneOf": [{"type": "boolean"}]}},
        }

        normalized = _normalize_schema_for_gemini(schema)

        assert normalized["properties"]["one_of_field"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
        assert normalized["properties"]["list_field"] == {"type": "array", "items": {"anyOf": [{"type": "string"}]}}
        assert normalized["anyOf"] == [{"properties": {"nested": {"anyOf": [{"type": "integer"}]}}}]
        assert normalized["$defs"] == {"inner": {"anyOf": [{"type": "boolean"}]}}
        assert normalized["definitions"] == {"inner2": {"anyOf": [{"type": "boolean"}]}}

    def test_exotic_keywords_stripped(self) -> None:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "allOf": [{"type": "object"}],
            "not": {"type": "string"},
            "const": "fixed",
            "examples": ["a", "b"],
            "discriminator": {"propertyName": "kind"},
            "type": "object",
        }

        normalized = _normalize_schema_for_gemini(schema)

        assert normalized == {"type": "object"}

    def test_plain_schema_passes_through_unchanged(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A name.", "enum": ["a", "b"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
        }
        original = copy.deepcopy(schema)

        normalized = _normalize_schema_for_gemini(schema)

        assert normalized == schema == original

    def test_non_dict_list_scalars_pass_through(self) -> None:
        assert _normalize_schema_for_gemini("string") == "string"
        assert _normalize_schema_for_gemini(42) == 42
        assert _normalize_schema_for_gemini(None) is None
        assert _normalize_schema_for_gemini(["a", 1, None]) == ["a", 1, None]
        assert _normalize_schema_for_gemini([]) == []


class TestToolsListIsolation:
    """Regression tests for the Gemini tools_list conversion path."""

    def _definition(self, name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": name, "description": f"{name} description", "parameters": parameters},
        }

    def test_oneof_tool_produces_valid_declaration(self) -> None:
        definition = self._definition("mcp_melious_rest_scrape_url", SCRAPE_URL_PARAMETERS)

        with patch.object(RegisteredChibiTools, "get_tool_definitions", return_value=[definition]):
            tools = Gemini(token=TEST_TOKEN).tools_list

        assert len(tools) == 1
        declarations = tools[0].function_declarations
        assert declarations is not None
        declaration = declarations[0]
        assert isinstance(declaration, FunctionDeclaration)
        assert declaration.name == "mcp_melious_rest_scrape_url"
        assert declaration.parameters is not None
        assert declaration.parameters.any_of is not None
        assert [branch.required for branch in declaration.parameters.any_of] == [["url"], ["urls"]]

    def test_unconvertible_tool_is_skipped_not_raised(self) -> None:
        still_exotic = {"type": "object", "unevaluatedProperties": False}
        good = self._definition("good_tool", {"type": "object", "properties": {}})
        bad = self._definition("bad_tool", still_exotic)

        with patch.object(RegisteredChibiTools, "get_tool_definitions", return_value=[bad, good]):
            tools = Gemini(token=TEST_TOKEN).tools_list

        offered_names = [t.function_declarations[0].name for t in tools if t.function_declarations]
        assert offered_names == ["good_tool"]

    def test_skipped_tool_logs_warning_naming_the_tool(self) -> None:
        bad = self._definition("bad_tool", {"type": "object", "unevaluatedProperties": False})
        warning = MagicMock()

        with (
            patch.object(RegisteredChibiTools, "get_tool_definitions", return_value=[bad]),
            patch("chibi.services.providers.gemini_native.logger") as logger_mock,
        ):
            logger_mock.warning = warning
            Gemini(token=TEST_TOKEN).tools_list

        assert warning.called
        assert "bad_tool" in warning.call_args.args[0]

    @pytest.mark.parametrize("parameters", [{"type": "object", "oneOf": [{"required": ["url"]}]}])
    def test_tools_list_never_raises_for_oneof(self, parameters: dict[str, Any]) -> None:
        definition = self._definition("tool_with_oneof", parameters)

        with patch.object(RegisteredChibiTools, "get_tool_definitions", return_value=[definition]):
            tools = Gemini(token=TEST_TOKEN).tools_list

        assert len(tools) == 1
