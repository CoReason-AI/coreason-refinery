# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

import pytest
from pydantic import ValidationError

from coreason_refinery.parsing import DocumentParser, MockParser, ParsedElement


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
