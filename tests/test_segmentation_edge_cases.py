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


def test_trailing_dots_numbering(chunker: SemanticChunker) -> None:
    """Test that '1.' and '1.1.' are handled correctly."""
    # "1." -> Matches "1" -> Depth 1
    # "1.1." -> Matches "1.1" -> Depth 2
    elements = [
        ParsedElement(text="Doc", type="TITLE"),
        ParsedElement(text="1. Introduction", type="HEADER"),
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT"),
        ParsedElement(text="1.1. Background", type="HEADER"),
        ParsedElement(text="Content 1.1", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 2

    # Chunk 0
    assert chunks[0].metadata["header_hierarchy"] == ["Doc", "1. Introduction"]

    # Chunk 1
    # Should nest 1.1 under 1
    assert chunks[1].metadata["header_hierarchy"] == ["Doc", "1. Introduction", "1.1. Background"]


def test_hierarchy_skipping(chunker: SemanticChunker) -> None:
    """Test skipping levels (e.g. 1 -> 1.1.1)."""
    elements = [
        ParsedElement(text="Root", type="TITLE"),
        ParsedElement(text="1. Top", type="HEADER"),
        # Skip 1.1, go straight to 1.1.1 (Depth 3)
        ParsedElement(text="1.1.1 Deep", type="HEADER"),
        ParsedElement(text="Content Deep", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    # Only one chunk is produced because intermediate headers have no content
    assert len(chunks) == 1
    # The hierarchy should reflect the skip naturally
    assert chunks[0].metadata["header_hierarchy"] == ["Root", "1. Top", "1.1.1 Deep"]


def test_mixed_numbered_and_unnumbered(chunker: SemanticChunker) -> None:
    """Test interaction between numbered sections and unnumbered 'semantic' headers."""
    elements = [
        ParsedElement(text="Root", type="TITLE"),
        ParsedElement(text="1. Intro", type="HEADER"),
        ParsedElement(text="Text", type="NARRATIVE_TEXT"),
        # "Methodology" has no number -> Depth 1 (default)
        # Should pop "1. Intro"
        ParsedElement(text="Methodology", type="HEADER"),
        ParsedElement(text="Text", type="NARRATIVE_TEXT"),
        # "2. Results" -> Depth 1
        # Should pop "Methodology"
        ParsedElement(text="2. Results", type="HEADER"),
        ParsedElement(text="2.1 Analysis", type="HEADER"),
        ParsedElement(text="Analysis Content", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 3

    assert chunks[0].metadata["header_hierarchy"] == ["Root", "1. Intro"]
    assert chunks[1].metadata["header_hierarchy"] == ["Root", "Methodology"]
    assert chunks[2].metadata["header_hierarchy"] == ["Root", "2. Results", "2.1 Analysis"]


def test_deeply_nested_numbering(chunker: SemanticChunker) -> None:
    """Test very deep nesting."""
    depth_str = "1.2.3.4.5.6.7"
    text = f"{depth_str} Abyss"
    depth = chunker._infer_depth(text)
    assert depth == 7

    elements = [
        ParsedElement(text="Root", type="TITLE"),
        ParsedElement(text=text, type="HEADER"),
        ParsedElement(text="Hello", type="NARRATIVE_TEXT"),
    ]
    chunks = chunker.chunk(elements)
    assert chunks[0].metadata["header_hierarchy"] == ["Root", text]


def test_whitespace_chaos(chunker: SemanticChunker) -> None:
    """Test erratic whitespace handling."""
    elements = [
        ParsedElement(text="   1.   Start", type="HEADER"),  # Matches "1" -> Depth 1
        ParsedElement(text="Content", type="NARRATIVE_TEXT"),
        ParsedElement(text="\t1.1\tSub", type="HEADER"),  # Matches "1.1" -> Depth 2
        ParsedElement(text="Content", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    assert chunks[0].metadata["header_hierarchy"] == ["   1.   Start"]
    assert chunks[1].metadata["header_hierarchy"] == ["   1.   Start", "\t1.1\tSub"]


def test_labeled_section_variants(chunker: SemanticChunker) -> None:
    """Test various labeled section formats."""
    # Section 1.2
    assert chunker._infer_depth("Section 1.2") == 2
    # PART III (Roman not supported yet, falls back to 1)
    assert chunker._infer_depth("PART III") == 1
    # APPENDIX 5.1
    assert chunker._infer_depth("APPENDIX 5.1") == 2
    # chapter 10
    assert chunker._infer_depth("chapter 10") == 1
