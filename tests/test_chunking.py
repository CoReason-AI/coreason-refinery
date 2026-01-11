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
from coreason_refinery.chunking import SemanticChunker
from coreason_refinery.models import IngestionConfig
from coreason_refinery.parsing import ParsedElement


@pytest.fixture
def chunker() -> SemanticChunker:
    config = IngestionConfig(chunk_strategy="HEADER", segment_len=500)
    return SemanticChunker(config)


def test_chunking_hierarchy(chunker: SemanticChunker) -> None:
    """Test that chunks correctly inherit context from headers."""
    elements = [
        ParsedElement(text="Protocol 123", type="TITLE", metadata={}),
        ParsedElement(text="Introduction", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="This is the intro.", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="Safety", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Adverse Events", type="HEADER", metadata={"section_depth": 2}),
        ParsedElement(text="Bad things happen.", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 1: Intro
    # Context: Protocol 123 > Introduction
    assert "Context: Protocol 123 > Introduction" in chunks[0].text
    assert "This is the intro." in chunks[0].text
    assert chunks[0].metadata["header_hierarchy"] == ["Protocol 123", "Introduction"]

    # Chunk 2: Safety > Adverse Events
    # Context: Protocol 123 > Safety > Adverse Events
    assert "Context: Protocol 123 > Safety > Adverse Events" in chunks[1].text
    assert "Bad things happen." in chunks[1].text
    assert chunks[1].metadata["header_hierarchy"] == [
        "Protocol 123",
        "Safety",
        "Adverse Events",
    ]


def test_chunking_implicit_depth(chunker: SemanticChunker) -> None:
    """Test that headers without explicit depth default to level 1."""
    elements = [
        ParsedElement(text="Root", type="TITLE", metadata={}),
        ParsedElement(text="Section A", type="HEADER", metadata={}),  # Default depth 1
        ParsedElement(text="Content A", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="Section B", type="HEADER", metadata={}),  # Default depth 1, should replace A
        ParsedElement(text="Content B", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2
    assert chunks[0].metadata["header_hierarchy"] == ["Root", "Section A"]
    assert chunks[1].metadata["header_hierarchy"] == ["Root", "Section B"]


def test_chunking_table_handling(chunker: SemanticChunker) -> None:
    """Test that tables are included in the chunks."""
    elements = [
        ParsedElement(text="Results", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="| A | B |", type="TABLE", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    assert "| A | B |" in chunks[0].text
    assert chunks[0].metadata["header_hierarchy"] == ["Results"]


def test_empty_input(chunker: SemanticChunker) -> None:
    """Test empty input list."""
    chunks = chunker.chunk([])
    assert chunks == []


def test_title_reset(chunker: SemanticChunker) -> None:
    """Test that a new TITLE resets the hierarchy."""
    elements = [
        ParsedElement(text="Doc 1", type="TITLE", metadata={}),
        ParsedElement(text="Sec 1", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="Doc 2", type="TITLE", metadata={}),  # Should clear Doc 1 and Sec 1
        ParsedElement(text="Content 2", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 1
    assert chunks[0].metadata["header_hierarchy"] == ["Doc 1", "Sec 1"]

    # Chunk 2
    assert chunks[1].metadata["header_hierarchy"] == ["Doc 2"]
    assert "Content 2" in chunks[1].text


def test_consecutive_headers(chunker: SemanticChunker) -> None:
    """Test that consecutive headers don't produce empty chunks."""
    elements = [
        ParsedElement(text="H1", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="H2", type="HEADER", metadata={"section_depth": 2}),
        ParsedElement(text="Content", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    # Should only produce one chunk for the content under H2.
    assert len(chunks) == 1
    assert chunks[0].metadata["header_hierarchy"] == ["H1", "H2"]
    assert "Content" in chunks[0].text


def test_footer_ignored(chunker: SemanticChunker) -> None:
    """Test that footers are ignored."""
    elements = [
        ParsedElement(text="H1", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Content", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="Page 1", type="FOOTER", metadata={}),
        ParsedElement(text="More Content", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    assert "Content" in chunks[0].text
    assert "Page 1" not in chunks[0].text
    assert "More Content" in chunks[0].text


def test_content_no_header(chunker: SemanticChunker) -> None:
    """Test content without any header context."""
    elements = [
        ParsedElement(text="Just some text", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="More text", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    assert chunks[0].text == "Just some text\n\nMore text"
    assert chunks[0].metadata["header_hierarchy"] == []
    assert "Context:" not in chunks[0].text


def test_complex_depth_jumps(chunker: SemanticChunker) -> None:
    """Test non-linear depth changes (skipping levels)."""
    elements = [
        ParsedElement(text="Root", type="TITLE", metadata={}),
        ParsedElement(text="H1", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="H3", type="HEADER", metadata={"section_depth": 3}),  # Skips depth 2
        ParsedElement(text="Content A", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="H2", type="HEADER", metadata={"section_depth": 2}),  # Back up to 2
        ParsedElement(text="Content B", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 1: Root > H1 > H3
    assert chunks[0].metadata["header_hierarchy"] == ["Root", "H1", "H3"]
    assert "Content A" in chunks[0].text

    # Chunk 2: Root > H1 > H2 (H3 should be popped, H1 retained as 1 < 2)
    assert chunks[1].metadata["header_hierarchy"] == ["Root", "H1", "H2"]
    assert "Content B" in chunks[1].text


def test_mixed_content_types(chunker: SemanticChunker) -> None:
    """Test mixing text, lists, and tables in one section."""
    elements = [
        ParsedElement(text="Section Mixed", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Intro text.", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="* Item 1", type="LIST_ITEM", metadata={}),
        ParsedElement(text="* Item 2", type="LIST_ITEM", metadata={}),
        ParsedElement(text="| Col1 | Col2 |", type="TABLE", metadata={}),
        ParsedElement(text="Outro text.", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    text = chunks[0].text
    assert "Context: Section Mixed" in text
    assert "Intro text." in text
    assert "* Item 1" in text
    assert "* Item 2" in text
    assert "| Col1 | Col2 |" in text
    assert "Outro text." in text
    # Verify order roughly
    assert text.index("Intro text.") < text.index("| Col1 | Col2 |")


def test_header_empty_text(chunker: SemanticChunker) -> None:
    """Test a header with empty text string."""
    elements = [
        ParsedElement(text="Root", type="TITLE", metadata={}),
        ParsedElement(text="", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Content", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    # Should be ["Root", ""]
    assert chunks[0].metadata["header_hierarchy"] == ["Root", ""]
    # Context string should probably look like "Context: Root > " or similar.
    assert "Context: Root > " in chunks[0].text


def test_title_in_middle(chunker: SemanticChunker) -> None:
    """Test a TITLE appearing in the middle of content."""
    elements = [
        ParsedElement(text="Doc 1", type="TITLE", metadata={}),
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="Doc 2", type="TITLE", metadata={}),
        ParsedElement(text="Content 2", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2
    assert chunks[0].metadata["header_hierarchy"] == ["Doc 1"]
    assert chunks[1].metadata["header_hierarchy"] == ["Doc 2"]
