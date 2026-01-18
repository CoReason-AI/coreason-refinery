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
    config = IngestionConfig(chunk_strategy="HEADER", segment_len=500)
    return SemanticChunker(config)


def test_depth_inference(chunker: SemanticChunker) -> None:
    """Test that header depth is inferred from numbering if metadata is missing."""
    elements = [
        ParsedElement(text="Root", type="TITLE", metadata={}),
        # "1. Section" -> Depth 1
        ParsedElement(text="1. Section A", type="HEADER", metadata={}),
        # "1.1 Subsection" -> Depth 2
        ParsedElement(text="1.1 Subsection B", type="HEADER", metadata={}),
        # "1.1.1 Detail" -> Depth 3
        ParsedElement(text="1.1.1 Detail C", type="HEADER", metadata={}),
        ParsedElement(text="Content", type="NARRATIVE_TEXT", metadata={}),
        # "2. Section D" -> Depth 1 (should pop back to root)
        ParsedElement(text="2. Section D", type="HEADER", metadata={}),
        ParsedElement(text="More Content", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 1: Root > 1. Section A > 1.1 Subsection B > 1.1.1 Detail C
    assert chunks[0].metadata["header_hierarchy"] == [
        "Root",
        "1. Section A",
        "1.1 Subsection B",
        "1.1.1 Detail C",
    ]
    assert "Content" in chunks[0].text

    # Chunk 2: Root > 2. Section D
    assert chunks[1].metadata["header_hierarchy"] == ["Root", "2. Section D"]
    assert "More Content" in chunks[1].text


def test_speaker_notes_injection(chunker: SemanticChunker) -> None:
    """Test that speaker notes are prepended to the chunk context."""
    elements = [
        ParsedElement(text="Slide Title", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(
            text="Bullet point 1",
            type="LIST_ITEM",
            metadata={"speaker_notes": "Don't forget to mention safety."},
        ),
        ParsedElement(
            text="Bullet point 2",
            type="LIST_ITEM",
            metadata={"speaker_notes": "Also mention efficacy."},
        ),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    text = chunks[0].text

    # Check that speaker notes appear
    assert "Speaker Notes: Don't forget to mention safety." in text
    assert "Speaker Notes: Also mention efficacy." in text

    assert "Bullet point 1" in text


def test_metadata_aggregation(chunker: SemanticChunker) -> None:
    """Test that metadata fields like page_number are aggregated."""
    elements = [
        ParsedElement(text="Header", type="HEADER", metadata={"page_number": 1}),
        ParsedElement(text="Text on page 1", type="NARRATIVE_TEXT", metadata={"page_number": 1}),
        ParsedElement(text="Text on page 2", type="NARRATIVE_TEXT", metadata={"page_number": 2}),
        ParsedElement(text="Text on page 2 again", type="NARRATIVE_TEXT", metadata={"page_number": 2}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    # Check unique page numbers
    assert "page_numbers" in chunks[0].metadata
    assert sorted(chunks[0].metadata["page_numbers"]) == [1, 2]

    # Check that header hierarchy is preserved in metadata
    assert "header_hierarchy" in chunks[0].metadata


def test_depth_inference_edge_cases(chunker: SemanticChunker) -> None:
    """Test edge cases for depth inference."""
    # No number -> Default 1
    assert chunker._infer_depth("Introduction") == 1

    # "1." -> 1
    assert chunker._infer_depth("1. Introduction") == 1

    # "1.2.3.4" -> 4
    assert chunker._infer_depth("1.2.3.4 Deep") == 4

    # "Appendix A" -> 1 (Non-numeric)
    assert chunker._infer_depth("Appendix A") == 1


def test_mixed_depth_sources(chunker: SemanticChunker) -> None:
    """Test mixing explicit section_depth metadata with inferred depth."""
    elements = [
        ParsedElement(text="Root", type="TITLE", metadata={}),
        # Explicit Depth 1
        ParsedElement(text="Explicit 1", type="HEADER", metadata={"section_depth": 1}),
        # Inferred Depth 2 ("1.1")
        ParsedElement(text="1.1 Inferred", type="HEADER", metadata={}),
        # Explicit Depth 3 (skips 2.1)
        ParsedElement(text="Explicit 3", type="HEADER", metadata={"section_depth": 3}),
        ParsedElement(text="Content", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    hierarchy = chunks[0].metadata["header_hierarchy"]
    assert hierarchy == ["Root", "Explicit 1", "1.1 Inferred", "Explicit 3"]


def test_header_regex_edge_cases(chunker: SemanticChunker) -> None:
    """Test _infer_depth with ambiguous inputs."""
    # Date-like header: "2023.01.01 Review"
    # Matches \d+(\.\d+)* -> "2023.01.01" -> 2 dots -> depth 3.
    # This is "correct" for the regex logic, even if semantically debatable.
    assert chunker._infer_depth("2023.01.01 Review") == 3

    # Quantity-like: "10.5 kg limit"
    # Matches "10.5" -> 1 dot -> depth 2.
    assert chunker._infer_depth("10.5 kg limit") == 2

    # Bullet-like? "1) Item" -> No match for ^\d+(\.\d+)* (unless we loosen regex)
    # The current regex expects dot separators.
    assert chunker._infer_depth("1) Item") == 1

    # Leading whitespace? " 1.2 Title"
    # Updated expectation: The new regex allows leading whitespace \s*
    assert chunker._infer_depth(" 1.2 Title") == 2

    # "Section 1.2"
    assert chunker._infer_depth("Section 1.2") == 2

    # "Chapter 3.1.1"
    assert chunker._infer_depth("Chapter 3.1.1") == 3


def test_disjoint_page_numbers(chunker: SemanticChunker) -> None:
    """Test aggregation of page numbers from non-sequential pages."""
    elements = [
        ParsedElement(text="H1", type="HEADER", metadata={"page_number": 1}),
        ParsedElement(text="A", type="NARRATIVE_TEXT", metadata={"page_number": 1}),
        ParsedElement(text="B", type="NARRATIVE_TEXT", metadata={"page_number": 10}),
        ParsedElement(text="C", type="NARRATIVE_TEXT", metadata={"page_number": 5}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    # Should be sorted unique list
    assert chunks[0].metadata["page_numbers"] == [1, 5, 10]


def test_hierarchy_recovery(chunker: SemanticChunker) -> None:
    """Test deep nesting popping back to shallow."""
    elements = [
        ParsedElement(text="Root", type="TITLE", metadata={}),
        ParsedElement(text="1", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="1.1", type="HEADER", metadata={"section_depth": 2}),
        ParsedElement(text="1.1.1", type="HEADER", metadata={"section_depth": 3}),
        ParsedElement(text="Deep Content", type="NARRATIVE_TEXT", metadata={}),
        # Pop back to 1
        ParsedElement(text="2", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Shallow Content", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 1: Root > 1 > 1.1 > 1.1.1
    assert chunks[0].metadata["header_hierarchy"] == ["Root", "1", "1.1", "1.1.1"]

    # Chunk 2: Root > 2
    assert chunks[1].metadata["header_hierarchy"] == ["Root", "2"]


def test_section_named_header_inference(chunker: SemanticChunker) -> None:
    """Test that headers like 'Section 1.1' are correctly inferred as depth 2."""
    elements = [
        ParsedElement(text="Protocol", type="TITLE"),
        ParsedElement(text="Section 1: Overview", type="HEADER"),
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT"),
        ParsedElement(text="Section 1.1: Details", type="HEADER"),
        ParsedElement(text="Content 1.1", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 0: Section 1
    # Expect: Protocol > Section 1
    assert chunks[0].metadata["header_hierarchy"] == ["Protocol", "Section 1: Overview"]

    # Chunk 1: Section 1.1
    # Expect: Protocol > Section 1 > Section 1.1
    assert chunks[1].metadata["header_hierarchy"] == ["Protocol", "Section 1: Overview", "Section 1.1: Details"]


def test_markdown_mixed_with_numbering(chunker: SemanticChunker) -> None:
    """Test that markdown takes precedence, but if absent, numbering works."""
    elements = [
        ParsedElement(text="Doc", type="TITLE"),
        ParsedElement(text="# Top Level", type="HEADER"),  # Markdown depth 1
        ParsedElement(text="1.1 Sub", type="HEADER"),  # Numbering depth 2
        ParsedElement(text="Content", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    # The header cleaner strips leading hashes, so "# Top Level" becomes "Top Level"
    assert chunks[0].metadata["header_hierarchy"] == ["Doc", "Top Level", "1.1 Sub"]
