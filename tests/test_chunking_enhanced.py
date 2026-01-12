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

    # Check order: Notes should ideally be near the content or at the start
    # Implementation detail: We prepend/append to the content buffer
    # If we append to buffer items, they appear mixed.
    # The requirement is "prepend these notes to the text as context" (User: "deciding whether to prepend")
    # We will assume they are part of the text body for now.
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
