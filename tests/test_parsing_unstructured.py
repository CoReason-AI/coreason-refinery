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


def test_metadata_preservation(parser: UnstructuredPdfParser) -> None:
    """Test that arbitrary metadata fields are preserved in the output."""
    metadata = ElementMetadata(
        page_number=5,
        filename="test.pdf",
        file_directory="/tmp",
        languages=["eng"],
    )
    # unstructured might add other fields, we check if they survive to_dict
    element = NarrativeText(text="Metadata Test", metadata=metadata)

    with patch("coreason_refinery.parsing.partition_pdf", return_value=[element]):
        results = parser.parse("test.pdf")

        meta_out = results[0].metadata
        assert meta_out["page_number"] == 5
        assert meta_out["filename"] == "test.pdf"
        assert meta_out["file_directory"] == "/tmp"
        assert meta_out["languages"] == ["eng"]


def test_element_ordering(parser: UnstructuredPdfParser) -> None:
    """Test that the order of elements is strictly preserved."""
    elements = [
        Title(text="First", metadata=ElementMetadata(page_number=1)),
        NarrativeText(text="Second", metadata=ElementMetadata(page_number=1)),
        Table(text="Third", metadata=ElementMetadata(page_number=2)),
        Footer(text="Fourth", metadata=ElementMetadata(page_number=2)),
    ]

    with patch("coreason_refinery.parsing.partition_pdf", return_value=elements):
        results = parser.parse("doc.pdf")

        assert len(results) == 4
        assert results[0].text == "First"
        assert results[1].text == "Second"
        assert results[2].text == "Third"
        assert results[3].text == "Fourth"


def test_complex_table_structure(parser: UnstructuredPdfParser) -> None:
    """Test a complex table with markdown representation."""
    table_text = "| Header 1 | Header 2 |\n| --- | --- |\n| Cell 1 | Cell 2 |"
    element = Table(text=table_text, metadata=ElementMetadata(page_number=10, text_as_html="<table>...</table>"))

    with patch("coreason_refinery.parsing.partition_pdf", return_value=[element]):
        results = parser.parse("doc.pdf")

        assert results[0].type == "TABLE"
        assert results[0].text == table_text
        # Verify that even 'text_as_html' if present in metadata is preserved
        assert results[0].metadata["text_as_html"] == "<table>...</table>"


def test_whitespace_and_empty_elements(parser: UnstructuredPdfParser) -> None:
    """Test handling of empty or whitespace-only elements."""
    elements = [
        NarrativeText(text="", metadata=ElementMetadata(page_number=1)),
        NarrativeText(text="   ", metadata=ElementMetadata(page_number=1)),
        NarrativeText(text="\n", metadata=ElementMetadata(page_number=1)),
    ]

    with patch("coreason_refinery.parsing.partition_pdf", return_value=elements):
        results = parser.parse("doc.pdf")

        assert len(results) == 3
        assert results[0].text == ""
        assert results[1].text == "   "
        assert results[2].text == "\n"
