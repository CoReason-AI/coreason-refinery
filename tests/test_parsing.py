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
from pydantic import ValidationError

from coreason_refinery.models import ParsedElement
from coreason_refinery.parsing import DocumentParser, ParserRegistry


# --- Mock Implementation ---
class MockParser(DocumentParser):
    def parse(self, file_path: str) -> List[ParsedElement]:
        return [
            ParsedElement(type="TITLE", text="Test Title"),
            ParsedElement(type="NARRATIVE_TEXT", text="Test content."),
        ]


# --- Tests for ParsedElement ---
def test_parsed_element_creation() -> None:
    """Test creating a valid ParsedElement."""
    element = ParsedElement(type="TITLE", text="Hello World", metadata={"page": 1})
    assert element.type == "TITLE"
    assert element.text == "Hello World"
    assert element.metadata["page"] == 1


def test_parsed_element_validation() -> None:
    """Test validation of ParsedElement types."""
    with pytest.raises(ValidationError):
        ParsedElement(type="INVALID_TYPE", text="content")  # type: ignore[arg-type]


# --- Tests for ParserRegistry ---


def test_registry_registration_and_retrieval() -> None:
    """Test registering and retrieving a parser."""
    ParserRegistry.register(".mock", MockParser)
    parser_cls = ParserRegistry.get_parser(".mock")
    assert parser_cls == MockParser


def test_registry_case_insensitivity() -> None:
    """Test that registry handles case-insensitive extensions."""
    ParserRegistry.register(".MOCK", MockParser)
    parser_cls = ParserRegistry.get_parser(".mock")
    assert parser_cls == MockParser

    ParserRegistry.register(".lowercase", MockParser)
    parser_cls_upper = ParserRegistry.get_parser(".LOWERCASE")
    assert parser_cls_upper == MockParser


def test_registry_missing_extension() -> None:
    """Test error when retrieving a parser for an unknown extension."""
    with pytest.raises(ValueError) as excinfo:
        ParserRegistry.get_parser(".unknown")
    assert "No parser registered for extension: .unknown" in str(excinfo.value)


def test_registry_overwrite() -> None:
    """Test overwriting an existing parser registration."""

    class AnotherMockParser(DocumentParser):
        def parse(self, file_path: str) -> List[ParsedElement]:
            return []

    ParserRegistry.register(".mock", MockParser)
    assert ParserRegistry.get_parser(".mock") == MockParser

    ParserRegistry.register(".mock", AnotherMockParser)
    assert ParserRegistry.get_parser(".mock") == AnotherMockParser


# --- Test MockParser ---
def test_mock_parser_parse() -> None:
    """Test the mock parser implementation."""
    parser = MockParser()
    elements = parser.parse("dummy.mock")
    assert len(elements) == 2
    assert elements[0].type == "TITLE"
    assert elements[1].text == "Test content."
