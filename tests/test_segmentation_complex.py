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


def test_the_report_structure(chunker: SemanticChunker) -> None:
    """Test complex report structure with deep nesting and returns."""
    # Structure: Title -> Executive Summary (H1) -> Background (H1) -> Problem (H2)
    # -> Solution (H2) -> Details (H3) -> Conclusion (H1)
    elements = [
        ParsedElement(text="Report Title", type="TITLE"),
        ParsedElement(text="# Executive Summary", type="HEADER"),
        ParsedElement(text="Summary content", type="NARRATIVE_TEXT"),
        ParsedElement(text="# Background", type="HEADER"),
        ParsedElement(text="Background content", type="NARRATIVE_TEXT"),
        ParsedElement(text="## Problem", type="HEADER"),
        ParsedElement(text="Problem content", type="NARRATIVE_TEXT"),
        ParsedElement(text="## Solution", type="HEADER"),
        ParsedElement(text="Solution content", type="NARRATIVE_TEXT"),
        ParsedElement(text="### Details", type="HEADER"),
        ParsedElement(text="Detail content", type="NARRATIVE_TEXT"),
        ParsedElement(text="# Conclusion", type="HEADER"),
        ParsedElement(text="Conclusion content", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    # Expected chunks correspond to content blocks:
    # 0. Summary content
    # 1. Background content
    # 2. Problem content
    # 3. Solution content
    # 4. Detail content
    # 5. Conclusion content
    assert len(chunks) == 6

    # 1. Executive Summary
    assert chunks[0].metadata["header_hierarchy"] == ["Report Title", "# Executive Summary"]

    # 2. Background (Should pop Exec Summary)
    assert chunks[1].metadata["header_hierarchy"] == ["Report Title", "# Background"]

    # 3. Problem (Should append to Background)
    assert chunks[2].metadata["header_hierarchy"] == ["Report Title", "# Background", "## Problem"]

    # 4. Solution (Should pop Problem, keep Background)
    assert chunks[3].metadata["header_hierarchy"] == ["Report Title", "# Background", "## Solution"]

    # 5. Details (Should append to Solution)
    assert chunks[4].metadata["header_hierarchy"] == [
        "Report Title",
        "# Background",
        "## Solution",
        "### Details",
    ]

    # 6. Conclusion (Should pop Details, Solution, Background. Keep Title)
    assert chunks[5].metadata["header_hierarchy"] == ["Report Title", "# Conclusion"]


def test_empty_section_nightmare(chunker: SemanticChunker) -> None:
    """Test handling of headers that have no immediate content."""
    # Title -> Chapter 1 (H1) -> [No Content] -> Section 1.1 (H2) -> Content.
    elements = [
        ParsedElement(text="Title", type="TITLE"),
        ParsedElement(text="# Chapter 1", type="HEADER"),
        # No content here
        ParsedElement(text="## Section 1.1", type="HEADER"),
        ParsedElement(text="Content", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    # Expectation:
    # Chunk 1: Content. Context: Title > Chapter 1 > Section 1.1
    # Note: Chapter 1 itself generates NO chunk because it has no content.

    assert len(chunks) == 1
    assert chunks[0].metadata["header_hierarchy"] == ["Title", "# Chapter 1", "## Section 1.1"]
    assert "Context: Title > # Chapter 1 > ## Section 1.1" in chunks[0].text


def test_deeply_nested_list(chunker: SemanticChunker) -> None:
    """Test merging of mixed content types under one header."""
    elements = [
        ParsedElement(text="# Header", type="HEADER"),
        ParsedElement(text="Intro", type="NARRATIVE_TEXT"),
        ParsedElement(text="Item 1", type="LIST_ITEM"),
        ParsedElement(text="Item 2", type="LIST_ITEM"),
        ParsedElement(text="| Table |", type="TABLE"),
        ParsedElement(text="Item 3", type="LIST_ITEM"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 1

    text = chunks[0].text
    assert "Intro" in text
    assert "Item 1" in text
    assert "| Table |" in text
    assert "Item 3" in text


def test_table_broken_by_header(chunker: SemanticChunker) -> None:
    """Negative Test: Table spanning page break interrupted by a new header."""
    # Table Part 1 -> Header (New Section) -> Table Part 2.
    # Should NOT merge.
    elements = [
        ParsedElement(text="# Section 1", type="HEADER"),
        ParsedElement(text="| Table Part 1 |", type="TABLE"),
        ParsedElement(text="# Section 2", type="HEADER"),
        ParsedElement(text="| Table Part 2 |", type="TABLE"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 2

    assert "| Table Part 1 |" in chunks[0].text
    assert "Context: # Section 1" in chunks[0].text

    assert "| Table Part 2 |" in chunks[1].text
    assert "Context: # Section 2" in chunks[1].text
