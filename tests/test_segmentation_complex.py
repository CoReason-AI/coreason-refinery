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
    assert chunks[0].metadata["header_hierarchy"] == ["Report Title", "Executive Summary"]

    # 2. Background (Should pop Exec Summary)
    assert chunks[1].metadata["header_hierarchy"] == ["Report Title", "Background"]

    # 3. Problem (Should append to Background)
    assert chunks[2].metadata["header_hierarchy"] == ["Report Title", "Background", "Problem"]

    # 4. Solution (Should pop Problem, keep Background)
    assert chunks[3].metadata["header_hierarchy"] == ["Report Title", "Background", "Solution"]

    # 5. Details (Should append to Solution)
    assert chunks[4].metadata["header_hierarchy"] == [
        "Report Title",
        "Background",
        "Solution",
        "Details",
    ]

    # 6. Conclusion (Should pop Details, Solution, Background. Keep Title)
    assert chunks[5].metadata["header_hierarchy"] == ["Report Title", "Conclusion"]


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
    assert chunks[0].metadata["header_hierarchy"] == ["Title", "Chapter 1", "Section 1.1"]
    assert "Context: Title > Chapter 1 > Section 1.1" in chunks[0].text


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
    assert "Context: Section 1" in chunks[0].text

    assert "| Table Part 2 |" in chunks[1].text
    assert "Context: Section 2" in chunks[1].text


def test_complex_mixed_attributes(chunker: SemanticChunker) -> None:
    """
    Test complex scenario with mixed attributes:
    - Deep nesting (Appendix A.1.2)
    - Table with Speaker Notes (Metadata injection)
    - Multiple tables in one section (Merged into one chunk)
    """
    elements = [
        ParsedElement(text="Research Report", type="TITLE"),
        ParsedElement(text="Appendix A", type="HEADER"),
        ParsedElement(text="A.1 Data", type="HEADER"),
        ParsedElement(text="A.1.2 Tables", type="HEADER"),
        # Content Start
        ParsedElement(text="Here is the data.", type="NARRATIVE_TEXT"),
        # Table with Speaker Notes
        ParsedElement(
            text="| ID | Value |\n| -- | -- |\n| 1 | 100 |",
            type="TABLE",
            metadata={"speaker_notes": "Emphasize the value 100 here."},
        ),
        # Another Table immediately following
        ParsedElement(text="| ID | Value |\n| -- | -- |\n| 2 | 200 |", type="TABLE"),
    ]

    chunks = chunker.chunk(elements)

    # Expectation:
    # Hierarchy: Research Report > Appendix A > A.1 Data > A.1.2 Tables
    # One single chunk containing: Narrative + Table 1 (w/ notes) + Table 2

    assert len(chunks) == 1
    chunk = chunks[0]

    # Check Hierarchy Context
    expected_hierarchy = ["Research Report", "Appendix A", "A.1 Data", "A.1.2 Tables"]
    assert chunk.metadata["header_hierarchy"] == expected_hierarchy
    assert "Context: Research Report > Appendix A > A.1 Data > A.1.2 Tables" in chunk.text

    # Check Content Merging
    assert "Here is the data." in chunk.text
    assert "| 1 | 100 |" in chunk.text
    assert "| 2 | 200 |" in chunk.text

    # Check Speaker Notes Injection
    # "Speaker Notes: Emphasize the value 100 here.\n| ID | Value |..."
    assert "Speaker Notes: Emphasize the value 100 here." in chunk.text


def test_edge_case_empty_title_and_orphaned_content(chunker: SemanticChunker) -> None:
    """
    Test edge cases:
    - Empty TITLE string.
    - Content before any header (Orphaned).
    """
    elements = [
        # Orphaned content at start (Should have no context prefix, or just own content)
        ParsedElement(text="Orphaned text.", type="NARRATIVE_TEXT"),
        # Empty Title (Should reset stack)
        ParsedElement(text="", type="TITLE"),
        # Header after empty title
        ParsedElement(text="Section 1", type="HEADER"),
        ParsedElement(text="Section content.", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    # Expectation:
    # Chunk 0: Orphaned text. Hierarchy: [].
    # Chunk 1: Section 1 content. Hierarchy: ["", "Section 1"] (Title is empty string)

    assert len(chunks) == 2

    # Chunk 0
    assert chunks[0].text.strip() == "Orphaned text."
    assert chunks[0].metadata["header_hierarchy"] == []

    # Chunk 1
    # Title text is empty string, so it effectively adds nothing visible to breadcrumbs if we just join
    # But it is in the stack.
    # Context:  > Section 1  (Because empty string joined).
    assert "Section content." in chunks[1].text
    assert chunks[1].metadata["header_hierarchy"] == ["", "Section 1"]

    # Verify context string formatting with empty title
    # "Context:  > Section 1"
    assert "Context:  > Section 1" in chunks[1].text
