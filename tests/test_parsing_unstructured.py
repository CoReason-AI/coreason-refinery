# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from unittest.mock import patch

import pytest
from coreason_refinery.parsing import UnstructuredPdfParser
from unstructured.documents.elements import (
    ElementMetadata,
    Footer,
    Header,
    ListItem,
    NarrativeText,
    Table,
    Title,
)


@pytest.fixture
def parser() -> UnstructuredPdfParser:
    return UnstructuredPdfParser()


def test_parse_pdf_success(parser: UnstructuredPdfParser) -> None:
    """Test successful parsing of a PDF with various element types."""

    # Mock return values from partition_pdf
    mock_elements = [
        Title(text="Study Protocol", metadata=ElementMetadata(page_number=1)),
        NarrativeText(text="This is the introduction.", metadata=ElementMetadata(page_number=1)),
        Table(text="| Col1 | Col2 |", metadata=ElementMetadata(page_number=2)),
        ListItem(text=" - Item 1", metadata=ElementMetadata(page_number=2)),
        Header(text="Header Content", metadata=ElementMetadata(page_number=3)),
        Footer(text="Footer Content", metadata=ElementMetadata(page_number=3)),
    ]

    with patch("coreason_refinery.parsing.partition_pdf", return_value=mock_elements) as mock_partition:
        results = parser.parse("dummy.pdf")

        mock_partition.assert_called_once_with(filename="dummy.pdf")
        assert len(results) == 6

        # Check Title mapping
        assert results[0].type == "TITLE"
        assert results[0].text == "Study Protocol"
        assert results[0].metadata["page_number"] == 1

        # Check NarrativeText mapping
        assert results[1].type == "NARRATIVE_TEXT"
        assert results[1].text == "This is the introduction."

        # Check Table mapping
        assert results[2].type == "TABLE"
        assert results[2].text == "| Col1 | Col2 |"

        # Check ListItem mapping
        assert results[3].type == "LIST_ITEM"

        # Check Header mapping
        assert results[4].type == "HEADER"

        # Check Footer mapping
        assert results[5].type == "FOOTER"


def test_parse_unknown_element(parser: UnstructuredPdfParser) -> None:
    """Test parsing of an unknown element type maps to UNCATEGORIZED."""

    # Better approach: Just use a raw unstructured Element which is the base class
    from unstructured.documents.elements import Element

    base_element = Element(element_id="123", coordinates=None, metadata=ElementMetadata())
    base_element.text = "Raw Element"

    with patch("coreason_refinery.parsing.partition_pdf", return_value=[base_element]):
        results = parser.parse("dummy.pdf")

        assert len(results) == 1
        assert results[0].type == "UNCATEGORIZED"
        assert results[0].text == "Raw Element"


def test_parse_empty_metadata(parser: UnstructuredPdfParser) -> None:
    """Test element with no metadata."""
    element = Title(text="No Metadata")
    # element.metadata is initialized empty by default in unstructured, not None.
    # But let's verify behavior.

    with patch("coreason_refinery.parsing.partition_pdf", return_value=[element]):
        results = parser.parse("dummy.pdf")
        assert results[0].metadata == {}


def test_parse_exception_propagation(parser: UnstructuredPdfParser) -> None:
    """Test that exceptions from partition_pdf are propagated."""
    with patch("coreason_refinery.parsing.partition_pdf", side_effect=IOError("File error")):
        with pytest.raises(IOError, match="File error"):
            parser.parse("bad_file.pdf")
