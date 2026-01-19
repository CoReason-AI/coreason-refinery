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


def test_skipping_levels(chunker: SemanticChunker) -> None:
    """Test behavior when hierarchy skips levels (H1 -> H3)."""
    elements = [
        ParsedElement(text="# H1", type="HEADER"),  # Depth 1
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT"),
        ParsedElement(text="### H3", type="HEADER"),  # Depth 3
        ParsedElement(text="Content 3", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 2

    # Chunk 1
    assert chunks[0].metadata["header_hierarchy"] == ["H1"]

    # Chunk 2
    # H3 should simply push onto H1
    assert chunks[1].metadata["header_hierarchy"] == ["H1", "H3"]


def test_backtracking_levels(chunker: SemanticChunker) -> None:
    """Test behavior when hierarchy backtracks (H1 -> H3 -> H2)."""
    elements = [
        ParsedElement(text="# H1", type="HEADER"),  # Depth 1
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT"),
        ParsedElement(text="### H3", type="HEADER"),  # Depth 3
        ParsedElement(text="Content 3", type="NARRATIVE_TEXT"),
        ParsedElement(text="## H2", type="HEADER"),  # Depth 2
        ParsedElement(text="Content 2", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 3

    # Chunk 3 (H2)
    # H2 (Depth 2) should pop H3 (Depth 3) but keep H1 (Depth 1)
    assert chunks[2].metadata["header_hierarchy"] == ["H1", "H2"]


def test_header_no_content_then_shallower(chunker: SemanticChunker) -> None:
    """Test a header with no content followed by a shallower header (H1 -> H2(empty) -> H1)."""
    elements = [
        ParsedElement(text="# H1 A", type="HEADER"),
        ParsedElement(text="Content A", type="NARRATIVE_TEXT"),
        ParsedElement(text="## H2 Empty", type="HEADER"),
        # No content
        ParsedElement(text="# H1 B", type="HEADER"),
        ParsedElement(text="Content B", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    # Chunk 1: Content A under H1 A
    # Chunk 2: Content B under H1 B
    # H2 Empty produces nothing and is popped by H1 B.

    assert len(chunks) == 2

    assert chunks[0].metadata["header_hierarchy"] == ["H1 A"]
    assert chunks[1].metadata["header_hierarchy"] == ["H1 B"]


def test_duplicate_headers_same_level(chunker: SemanticChunker) -> None:
    """Test consecutive headers at the same level."""
    elements = [
        ParsedElement(text="# H1 A", type="HEADER"),
        ParsedElement(text="Content A", type="NARRATIVE_TEXT"),
        ParsedElement(text="# H1 B", type="HEADER"),
        ParsedElement(text="Content B", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)
    assert len(chunks) == 2

    # H1 B should pop H1 A
    assert chunks[1].metadata["header_hierarchy"] == ["H1 B"]
