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


def test_unicode_and_special_characters(chunker: SemanticChunker) -> None:
    """Test handling of headers with Unicode and special characters."""
    elements = [
        ParsedElement(text="# Project 🚀 Omega", type="TITLE"),
        ParsedElement(text="## Section 1: 日本語", type="HEADER"),
        ParsedElement(text="Content with emojis 😃.", type="NARRATIVE_TEXT"),
        ParsedElement(text="## Section 2: <Tags>", type="HEADER"),
        ParsedElement(text="Content with tags.", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    # Expectation: 2 chunks
    assert len(chunks) == 2

    # Chunk 1
    assert "Context: # Project 🚀 Omega > ## Section 1: 日本語" in chunks[0].text
    assert "Content with emojis 😃." in chunks[0].text

    # Chunk 2
    assert "Context: # Project 🚀 Omega > ## Section 2: <Tags>" in chunks[1].text


def test_markdown_variations_and_whitespace(chunker: SemanticChunker) -> None:
    """Test messy Markdown headers and whitespace handling."""
    elements = [
        ParsedElement(text="   #   Messy Header   ", type="TITLE"),
        # Standard markdown allows trailing hashes
        ParsedElement(text="## Clean Header ##", type="HEADER"),
        ParsedElement(text="Content.", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    # Chunk 1
    # Note: Regex `^\s*(#+)` matches the starting hashes.
    # The text preserved in the chunk is the full text of the element.
    # The depth should be 2 for "## Clean Header ##".

    assert len(chunks) == 1
    # Check context formation
    # "   #   Messy Header   " should be depth 1 based on regex `^\s*(#+)` -> match `#` -> len 1.
    # Wait, TITLE is always forced to Depth 0 in logic.

    assert chunks[0].metadata["header_hierarchy"] == ["   #   Messy Header   ", "## Clean Header ##"]


def test_missing_optional_metadata(chunker: SemanticChunker) -> None:
    """Test graceful handling of elements with missing or None metadata."""
    elements = [
        ParsedElement(text="Title", type="TITLE", metadata={}),
        ParsedElement(text="# Header", type="HEADER", metadata={}),  # No page number, no depth
        ParsedElement(text="Content", type="NARRATIVE_TEXT", metadata={}),  # Empty metadata
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    assert chunks[0].text.endswith("Content")
    assert chunks[0].metadata["header_hierarchy"] == ["Title", "# Header"]
    # Should not crash looking for keys


def test_header_depth_extremes(chunker: SemanticChunker) -> None:
    """Test very deep nesting."""
    elements = [ParsedElement(text="Root", type="TITLE")]

    # Create depth 1 to 10
    expected_hierarchy = ["Root"]
    for i in range(1, 11):
        hashes = "#" * i
        text = f"{hashes} Level {i}"
        elements.append(ParsedElement(text=text, type="HEADER"))
        expected_hierarchy.append(text)

    elements.append(ParsedElement(text="Deep Content", type="NARRATIVE_TEXT"))

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    assert chunks[0].metadata["header_hierarchy"] == expected_hierarchy
    # Check that context string is constructed without error for long chain
    assert "Context: Root > # Level 1 > ## Level 2" in chunks[0].text


def test_alphanumeric_inference_robustness(chunker: SemanticChunker) -> None:
    """Verify inference robustness for trickier alphanumeric cases."""
    # "A.1" -> Depth 2 (Matches)
    # "A" -> Depth 1 (Fallback, does NOT match alphanumeric regex requiring dot)
    # "1.2.3.4" -> Depth 4

    elements = [
        ParsedElement(text="Doc", type="TITLE"),
        ParsedElement(text="A. Introduction", type="HEADER"),  # Should be Depth 1 (Fallback)
        ParsedElement(text="A.1 Subsection", type="HEADER"),  # Should be Depth 2
        ParsedElement(text="Content A.1", type="NARRATIVE_TEXT"),
        ParsedElement(text="B. Next Section", type="HEADER"),  # Should be Depth 1 (Fallback) - Pops A.1 and A.
        ParsedElement(text="Content B", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 2

    # Chunk 1: A.1
    # Stack: Doc -> A. -> A.1
    assert chunks[0].metadata["header_hierarchy"] == ["Doc", "A. Introduction", "A.1 Subsection"]

    # Chunk 2: B.
    # Stack: Doc -> B.
    # "B. Next Section" -> matches `^\s*([A-Z0-9]+(?:\.[A-Z0-9]+)+)`?
    # "B." -> "B" is group 1. dot is literal.
    # The regex `([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+)` expects repeating `.` then char.
    # "B." has a trailing dot. Does it match `(?:\.[A-Za-z0-9]+)`? No, it expects alphanumeric AFTER dot.
    # So "B." fails alphanumeric regex. Falls back to... 1?
    # Wait, "B." fallback?
    # It hits `match_digit`? No.
    # Hits default fallback -> 1.

    # So "B." is Depth 1.
    # Previous Stack Top: "A.1" (Depth 2).
    # 1 <= 2. Pop "A.1".
    # Next Stack Top: "A." (Depth 1).
    # 1 <= 1. Pop "A.".
    # Next Stack Top: "Doc" (Depth 0).
    # 1 > 0. Push "B.".

    assert chunks[1].metadata["header_hierarchy"] == ["Doc", "B. Next Section"]
