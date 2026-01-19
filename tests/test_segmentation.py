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
from coreason_refinery.models import IngestionConfig
from coreason_refinery.parsing import ParsedElement
from coreason_refinery.segmentation import SemanticChunker


@pytest.fixture
def chunker() -> SemanticChunker:
    config = IngestionConfig()
    return SemanticChunker(config)


def test_hierarchy_stacking(chunker: SemanticChunker) -> None:
    """Test that headers correctly push and pop from the context stack."""
    elements = [
        ParsedElement(text="Protocol", type="TITLE"),
        ParsedElement(text="1. Introduction", type="HEADER"),
        ParsedElement(text="Text 1", type="NARRATIVE_TEXT"),
        ParsedElement(text="1.1 Background", type="HEADER"),
        ParsedElement(text="Text 1.1", type="NARRATIVE_TEXT"),
        ParsedElement(text="2. Methods", type="HEADER"),
        ParsedElement(text="Text 2", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    # Expectation:
    # Chunk 0: Text 1. Context: Protocol > 1. Introduction
    # Chunk 1: Text 1.1. Context: Protocol > 1. Introduction > 1.1 Background
    # Chunk 2: Text 2. Context: Protocol > 2. Methods (1.1 and 1. should be popped)

    assert len(chunks) == 3

    # Check Chunk 0
    assert chunks[0].metadata["header_hierarchy"] == ["Protocol", "1. Introduction"]
    assert "Context: Protocol > 1. Introduction" in chunks[0].text
    assert "Text 1" in chunks[0].text

    # Check Chunk 1
    assert chunks[1].metadata["header_hierarchy"] == ["Protocol", "1. Introduction", "1.1 Background"]
    assert "Context: Protocol > 1. Introduction > 1.1 Background" in chunks[1].text
    assert "Text 1.1" in chunks[1].text

    # Check Chunk 2
    # "2. Methods" (Depth 1) should pop "1.1" (Depth 2) AND "1. Introduction" (Depth 1)
    # Stack should be ["Protocol", "2. Methods"]
    assert chunks[2].metadata["header_hierarchy"] == ["Protocol", "2. Methods"]
    assert "Context: Protocol > 2. Methods" in chunks[2].text
    assert "Text 2" in chunks[2].text


def test_table_rescue(chunker: SemanticChunker) -> None:
    """Test Story A: Table spanning elements merged into one block."""
    elements = [
        ParsedElement(text="1. Results", type="HEADER"),
        ParsedElement(text="| Col1 |", type="TABLE"),
        ParsedElement(text="| Val1 |", type="TABLE"),  # Simulating split table part 2
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    chunk_text = chunks[0].text

    # Verify both table parts are present
    assert "| Col1 |" in chunk_text
    assert "| Val1 |" in chunk_text
    # Verify hierarchy
    assert "Context: 1. Results" in chunk_text


def test_speaker_notes(chunker: SemanticChunker) -> None:
    """Test extraction and prepending of speaker notes."""
    elements = [
        ParsedElement(text="Slide 1", type="HEADER"),
        ParsedElement(
            text="Bullet Point 1",
            type="LIST_ITEM",
            metadata={"speaker_notes": "Don't forget to mention X"},
        ),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    assert "Speaker Notes: Don't forget to mention X" in chunks[0].text
    assert "Bullet Point 1" in chunks[0].text


def test_title_reset(chunker: SemanticChunker) -> None:
    """Test that a new TITLE resets the hierarchy stack completely."""
    elements = [
        ParsedElement(text="Old Doc", type="TITLE"),
        ParsedElement(text="Header 1", type="HEADER"),
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT"),
        ParsedElement(text="New Doc", type="TITLE"),
        ParsedElement(text="Header A", type="HEADER"),
        ParsedElement(text="Content A", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 0 from Old Doc
    assert chunks[0].metadata["header_hierarchy"] == ["Old Doc", "Header 1"]

    # Chunk 1 from New Doc
    # Should NOT contain Old Doc or Header 1
    assert "Old Doc" not in chunks[1].metadata["header_hierarchy"]
    assert chunks[1].metadata["header_hierarchy"] == ["New Doc", "Header A"]


def test_infer_depth_logic(chunker: SemanticChunker) -> None:
    """Test logic for inferring depth from various patterns."""

    # 1. Markdown
    assert chunker._infer_depth("# Heading 1") == 1
    assert chunker._infer_depth("## Heading 2") == 2
    assert chunker._infer_depth("### Heading 3") == 3

    # 2. Labeled
    assert chunker._infer_depth("Section 1") == 1
    assert chunker._infer_depth("Chapter 2.1") == 2
    assert chunker._infer_depth("Part 3.1.1") == 3
    assert chunker._infer_depth("Appendix A.1") == 2  # 1 dot -> 2

    # 3. Plain Numbering
    assert chunker._infer_depth("1. Top") == 1
    assert chunker._infer_depth("1.2 Sub") == 2
    assert chunker._infer_depth("1.2.3 Deep") == 3

    # 4. Fallback
    assert chunker._infer_depth("Introduction") == 1


def test_no_header_doc(chunker: SemanticChunker) -> None:
    """Test behavior with a document that has no headers."""
    elements = [
        ParsedElement(text="Just some text", type="NARRATIVE_TEXT"),
        ParsedElement(text="More text", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    # Should produce 1 chunk with all text, empty context
    assert len(chunks) == 1
    assert chunks[0].metadata["header_hierarchy"] == []
    assert "Just some text" in chunks[0].text
    assert "More text" in chunks[0].text
    # No "Context: " prefix if hierarchy is empty
    assert "Context: " not in chunks[0].text


def test_explicit_depth_override(chunker: SemanticChunker) -> None:
    """Test that metadata 'section_depth' overrides inference."""
    elements = [
        ParsedElement(text="Root", type="TITLE"),
        ParsedElement(text="First", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Second", type="HEADER", metadata={"section_depth": 2}),
        ParsedElement(text="Content", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    assert chunks[0].metadata["header_hierarchy"] == ["Root", "First", "Second"]


def test_chunk_page_numbers_aggregation(chunker: SemanticChunker) -> None:
    """Test that page numbers are aggregated from all elements in a chunk."""
    elements = [
        ParsedElement(text="H1", type="HEADER"),
        ParsedElement(text="P1", type="NARRATIVE_TEXT", metadata={"page_number": 1}),
        ParsedElement(text="P2", type="NARRATIVE_TEXT", metadata={"page_number": 2}),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 1
    # Check that both page 1 and 2 are recorded
    assert chunks[0].metadata["page_numbers"] == [1, 2]


def test_footer_ignored(chunker: SemanticChunker) -> None:
    """Test that footer elements are ignored."""
    elements = [
        ParsedElement(text="Header", type="HEADER"),
        ParsedElement(text="Content", type="NARRATIVE_TEXT"),
        ParsedElement(text="Footer Text", type="FOOTER"),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    assert "Footer Text" not in chunks[0].text


def test_mixed_hierarchy(chunker: SemanticChunker) -> None:
    """Test mixed hierarchy types (Markdown + Labeled)."""
    elements = [
        ParsedElement(text="# Root", type="HEADER"),  # Depth 1 (Markdown)
        ParsedElement(text="Section 1.1", type="HEADER"),  # Depth 2 (Labeled)
        ParsedElement(text="Content", type="NARRATIVE_TEXT"),
        ParsedElement(text="## Branch", type="HEADER"),  # Depth 2 (Markdown)
        ParsedElement(text="More Content", type="NARRATIVE_TEXT"),  # Added content so Chunk 2 is created
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 2

    # Chunk 0
    # Cleaned header expectations:
    assert chunks[0].metadata["header_hierarchy"] == ["Root", "Section 1.1"]

    # Chunk 1
    # ## Branch (Depth 2) should pop Section 1.1 (Depth 2) but keep # Root (Depth 1)
    # Cleaned header expectations:
    assert chunks[1].metadata["header_hierarchy"] == ["Root", "Branch"]
