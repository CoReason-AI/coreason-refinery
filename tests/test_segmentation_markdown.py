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


def test_markdown_header_depth(chunker: SemanticChunker) -> None:
    """Test that Markdown headers (#, ##) are correctly inferred for depth."""
    elements = [
        ParsedElement(text="# Title", type="TITLE"),
        # Level 1
        ParsedElement(text="# Section 1", type="HEADER"),
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT"),
        # Level 2
        ParsedElement(text="## Subsection 1.1", type="HEADER"),
        ParsedElement(text="Content 1.1", type="NARRATIVE_TEXT"),
        # Level 3
        ParsedElement(text="### Detail 1.1.1", type="HEADER"),
        ParsedElement(text="Content 1.1.1", type="NARRATIVE_TEXT"),
        # Pop back to Level 1
        ParsedElement(text="# Section 2", type="HEADER"),
        ParsedElement(text="Content 2", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 4
    # Note: Header cleaning removes leading # from the stored hierarchy strings
    assert chunks[0].metadata["header_hierarchy"] == ["Title", "Section 1"]
    assert chunks[1].metadata["header_hierarchy"] == ["Title", "Section 1", "Subsection 1.1"]
    assert chunks[2].metadata["header_hierarchy"] == ["Title", "Section 1", "Subsection 1.1", "Detail 1.1.1"]
    assert chunks[3].metadata["header_hierarchy"] == ["Title", "Section 2"]


def test_markdown_edge_cases(chunker: SemanticChunker) -> None:
    """Test various edge cases for Markdown header inference."""
    elements = [
        ParsedElement(text="Root", type="TITLE"),
        # Leading whitespace: "  ##" -> Depth 2
        ParsedElement(text="  ## Indented Header", type="HEADER"),
        ParsedElement(text="Content A", type="NARRATIVE_TEXT"),
        # Conflicting indicators: "# 1.1" -> Depth 1 (Markdown wins)
        ParsedElement(text="# 1.1 Conflicting", type="HEADER"),
        ParsedElement(text="Content B", type="NARRATIVE_TEXT"),
        # Trailing hashes: "## Header ##" -> Depth 2
        ParsedElement(text="## Trailing ##", type="HEADER"),
        ParsedElement(text="Content C", type="NARRATIVE_TEXT"),
        # Deep nesting: "######" -> Depth 6
        ParsedElement(text="###### Deep", type="HEADER"),
        ParsedElement(text="Content D", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 4

    # Chunk 0: Root > Indented Header (Depth 2)
    assert chunks[0].metadata["header_hierarchy"] == ["Root", "Indented Header"]

    # Chunk 1: Root > Conflicting (Depth 1)
    # Depth 1 pops the previous Depth 2
    assert chunks[1].metadata["header_hierarchy"] == ["Root", "1.1 Conflicting"]

    # Chunk 2: Root > Conflicting > Trailing (Depth 2)
    assert chunks[2].metadata["header_hierarchy"] == ["Root", "1.1 Conflicting", "Trailing ##"]

    # Chunk 3: Root > Conflicting > Trailing > Deep (Depth 6)
    assert chunks[3].metadata["header_hierarchy"] == [
        "Root",
        "1.1 Conflicting",
        "Trailing ##",
        "Deep",
    ]


def test_complex_hierarchy_mix(chunker: SemanticChunker) -> None:
    """Test mixing Markdown headers with non-Markdown headers."""
    elements = [
        ParsedElement(text="Doc Root", type="TITLE"),
        # Markdown Depth 1
        ParsedElement(text="# Main Section", type="HEADER"),
        # Numbered Depth 2
        ParsedElement(text="1.1 Sub Section", type="HEADER"),
        # Markdown Depth 3
        ParsedElement(text="### Deep Detail", type="HEADER"),
        # Numbered Depth 2 (Should pop Depth 3)
        ParsedElement(text="1.2 Another Sub", type="HEADER"),
        # Markdown Depth 1 (Should pop back to Root)
        ParsedElement(text="# Conclusion", type="HEADER"),
    ]

    chunks = chunker.chunk(elements)
    # The chunker creates chunks when it encounters a NEW header (flushing previous buffer)
    # or at the end.
    # Since we have no content between headers, it might create empty chunks or just move the stack.
    # Let's add content to ensure chunks are created.
    elements_with_content = []
    for e in elements:
        elements_with_content.append(e)
        elements_with_content.append(ParsedElement(text="Content", type="NARRATIVE_TEXT"))

    chunks = chunker.chunk(elements_with_content)
    assert len(chunks) == 6

    # Chunk 0: Content under "Doc Root"
    assert chunks[0].metadata["header_hierarchy"] == ["Doc Root"]

    # Chunk 1: Content under "# Main Section"
    assert chunks[1].metadata["header_hierarchy"] == ["Doc Root", "Main Section"]

    # Chunk 2: Content under "1.1 Sub Section"
    assert chunks[2].metadata["header_hierarchy"] == ["Doc Root", "Main Section", "1.1 Sub Section"]

    # Chunk 3: Content under "### Deep Detail"
    assert chunks[3].metadata["header_hierarchy"] == [
        "Doc Root",
        "Main Section",
        "1.1 Sub Section",
        "Deep Detail",
    ]

    # Chunk 4: Content under "1.2 Another Sub"
    # "1.2" (Depth 2) pops "###" (Depth 3)
    assert chunks[4].metadata["header_hierarchy"] == ["Doc Root", "Main Section", "1.2 Another Sub"]

    # Chunk 5: Content under "# Conclusion"
    # "# Conclusion" (Depth 1) pops "1.2" (Depth 2) and "# Main Section" (Depth 1)
    assert chunks[5].metadata["header_hierarchy"] == ["Doc Root", "Conclusion"]
