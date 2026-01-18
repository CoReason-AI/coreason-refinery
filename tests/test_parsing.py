# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from typing import List

import pytest
from coreason_refinery.parsing import DocumentParser, ParsedElement
from pydantic import ValidationError

from tests.mocks import MockParser


def test_parsed_element_creation() -> None:
    """Test creating a valid ParsedElement."""
    element = ParsedElement(text="Hello World", type="NARRATIVE_TEXT", metadata={"page": 1})
    assert element.text == "Hello World"
    assert element.type == "NARRATIVE_TEXT"
    assert element.metadata == {"page": 1}


def test_parsed_element_defaults() -> None:
    """Test ParsedElement defaults."""
    element = ParsedElement(text="Header", type="TITLE")
    assert element.metadata == {}


def test_parsed_element_validation() -> None:
    """Test ParsedElement validation."""
    with pytest.raises(ValidationError):
        # Invalid type
        ParsedElement(text="Test", type="INVALID_TYPE")  # type: ignore[arg-type]


def test_mock_parser_implementation() -> None:
    """Test that MockParser implements DocumentParser and returns expected data."""
    parser = MockParser()
    assert isinstance(parser, DocumentParser)

    elements = parser.parse("dummy/path.pdf")
    assert isinstance(elements, list)
    assert len(elements) == 4

    assert elements[0].type == "TITLE"
    assert elements[0].text == "# Clinical Protocol"

    assert elements[3].type == "TABLE"
    assert "| Dose |" in elements[3].text
    assert elements[3].metadata["is_table"] is True


def test_document_parser_abc() -> None:
    """Test that DocumentParser cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DocumentParser()  # type: ignore[abstract]


# --- Edge Case & Complex Scenario Tests ---


def test_parsed_element_edge_cases() -> None:
    """Test edge cases for ParsedElement (empty text, huge text)."""
    # Empty string text
    empty_element = ParsedElement(text="", type="NARRATIVE_TEXT")
    assert empty_element.text == ""

    # Whitespace text
    ws_element = ParsedElement(text="   ", type="NARRATIVE_TEXT")
    assert ws_element.text == "   "

    # Large text payload
    large_text = "a" * 10000
    large_element = ParsedElement(text=large_text, type="NARRATIVE_TEXT")
    assert len(large_element.text) == 10000


def test_parsed_element_complex_metadata() -> None:
    """Test ParsedElement with complex nested metadata."""
    complex_meta = {
        "page_info": {"number": 1, "orientation": "landscape"},
        "tags": ["urgent", "confidential"],
        "confidence_score": 0.98,
        "nested_list": [{"id": 1}, {"id": 2}],
    }
    element = ParsedElement(text="Complex Data", type="UNCATEGORIZED", metadata=complex_meta)

    assert element.metadata["page_info"]["orientation"] == "landscape"
    assert element.metadata["tags"][0] == "urgent"
    assert element.metadata["nested_list"][1]["id"] == 2


class SimpleLineParser(DocumentParser):
    """A concrete parser implementation for testing dynamic logic."""

    def parse(self, file_path: str) -> List[ParsedElement]:
        # Mimic reading a file by treating the file_path as the content for this test
        lines = file_path.split("\n")
        return [
            ParsedElement(text=line, type="NARRATIVE_TEXT", metadata={"line_num": i})
            for i, line in enumerate(lines)
            if line.strip()
        ]


def test_custom_parser_logic() -> None:
    """Verify that a custom implementation of the ABC works correctly."""
    parser = SimpleLineParser()
    # We pass the content as the path just for this simple test utility
    content = "Line 1\nLine 2\n\nLine 3"
    elements = parser.parse(content)

    assert len(elements) == 3
    assert elements[0].text == "Line 1"
    assert elements[0].metadata["line_num"] == 0
    assert elements[2].text == "Line 3"
    assert elements[2].metadata["line_num"] == 3  # 0-indexed, skipped line 2


class BrokenParser(DocumentParser):
    """A parser that simulates a failure."""

    def parse(self, file_path: str) -> List[ParsedElement]:
        raise FileNotFoundError(f"File not found: {file_path}")


def test_parser_error_propagation() -> None:
    """Verify that exceptions raised by concrete parsers propagate correctly."""
    parser = BrokenParser()
    with pytest.raises(FileNotFoundError) as excinfo:
        parser.parse("missing.pdf")

    assert "File not found: missing.pdf" in str(excinfo.value)
