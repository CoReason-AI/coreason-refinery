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


def test_complex_table_merge_across_pages(chunker: SemanticChunker) -> None:
    """Test 'The Table Rescue': Merging table parts across pages into one chunk.

    Scenario:
        1. Header 1 (Introduction)
        2. Text
        3. Header 2 (Dosing Schedule)
        4. Table Part 1 (Page 10)
        5. Table Part 2 (Page 11)
        6. Table Part 3 (Page 12)
        7. Header 2.1 (Notes)

    Expected:
        - Chunk 1: Header 1 + Text.
        - Chunk 2: Header 2 + Table Part 1 + Table Part 2 + Table Part 3.
        - Chunk 2 Metadata: page_numbers=[10, 11, 12].
    """
    elements = [
        ParsedElement(text="Protocol", type="TITLE", metadata={"page_number": 1}),
        ParsedElement(text="1. Introduction", type="HEADER", metadata={"page_number": 1}),
        ParsedElement(text="Intro text.", type="NARRATIVE_TEXT", metadata={"page_number": 1}),
        ParsedElement(text="2. Dosing Schedule", type="HEADER", metadata={"page_number": 10}),
        # Table spanning 3 pages
        ParsedElement(text="| Day | Dose |", type="TABLE", metadata={"page_number": 10, "table_id": "tbl1"}),
        ParsedElement(text="| 1 | 10mg |", type="TABLE", metadata={"page_number": 11, "table_id": "tbl1"}),
        ParsedElement(text="| 2 | 20mg |", type="TABLE", metadata={"page_number": 12, "table_id": "tbl1"}),
        # New section breaks the table chunk
        ParsedElement(text="2.1 Notes", type="HEADER", metadata={"page_number": 12}),
        ParsedElement(text="Check vitals.", type="NARRATIVE_TEXT", metadata={"page_number": 12}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 3

    # Check Table Chunk (Chunk 1 - index 1)
    table_chunk = chunks[1]

    # Context should be: Protocol > 2. Dosing Schedule
    expected_context = "Context: Protocol > 2. Dosing Schedule"
    assert table_chunk.text.startswith(expected_context)

    # Check that all table parts are present and joined
    assert "| Day | Dose |" in table_chunk.text
    assert "| 1 | 10mg |" in table_chunk.text
    assert "| 2 | 20mg |" in table_chunk.text

    # Verify Page Number Aggregation
    assert table_chunk.metadata["page_numbers"] == [10, 11, 12]

    # Verify Hierarchy
    assert table_chunk.metadata["header_hierarchy"] == ["Protocol", "2. Dosing Schedule"]


def test_complex_hierarchy_with_speaker_notes_and_depth_changes(chunker: SemanticChunker) -> None:
    """Test a complex flow with changing depths, speaker notes, and mixed numbering.

    Scenario:
        1. Title (Root)
        2. # Header 1 (Markdown Depth 1)
        3. Text with Speaker Notes
        4. Section 1.1 (Labeled Depth 2)
        5. Text
        6. 1.1.1 (Numbered Depth 3)
        7. Text
        8. ## Header 2 (Markdown Depth 2) - Should pop 1.1.1 and 1.1, but stay under Root?
           Wait: # Header 1 is Depth 1. ## Header 2 is Depth 2.
           So logic: New Depth 2. Stack was [Root(0), H1(1), H1.1(2), H1.1.1(3)].
           Pop H1.1.1 (3>=2). Pop H1.1 (2>=2).
           Stack becomes [Root(0), H1(1), H2(2)].
           Context: Root > H1 > H2.
    """
    elements = [
        ParsedElement(text="Presentation", type="TITLE"),
        # Depth 1
        ParsedElement(text="# Overview", type="HEADER"),
        ParsedElement(text="Slide Content", type="NARRATIVE_TEXT", metadata={"speaker_notes": "Say hello."}),
        # Depth 2
        ParsedElement(text="Section 1.1: Details", type="HEADER"),
        ParsedElement(text="Detail Content", type="NARRATIVE_TEXT"),
        # Depth 3
        ParsedElement(text="1.1.1 Deep Dive", type="HEADER"),
        ParsedElement(text="Deep Content", type="NARRATIVE_TEXT"),
        # Depth 2 (Markdown) - Should nest under # Overview (Depth 1)
        ParsedElement(text="## Summary", type="HEADER"),
        ParsedElement(text="Summary Text", type="NARRATIVE_TEXT"),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 4

    # Chunk 0: Overview
    assert chunks[0].metadata["header_hierarchy"] == ["Presentation", "# Overview"]
    assert "Speaker Notes: Say hello." in chunks[0].text

    # Chunk 1: Section 1.1
    assert chunks[1].metadata["header_hierarchy"] == ["Presentation", "# Overview", "Section 1.1: Details"]

    # Chunk 2: 1.1.1
    assert chunks[2].metadata["header_hierarchy"] == [
        "Presentation",
        "# Overview",
        "Section 1.1: Details",
        "1.1.1 Deep Dive",
    ]

    # Chunk 3: Summary (Depth 2)
    # Stack transition: [0, 1, 2, 3] -> New Depth 2 -> Pop 3, Pop 2 -> Push 2.
    # Result Stack: [0, 1, 2] -> Presentation > Overview > Summary
    assert chunks[3].metadata["header_hierarchy"] == ["Presentation", "# Overview", "## Summary"]

    assert "Context: Presentation > # Overview > ## Summary" in chunks[3].text
