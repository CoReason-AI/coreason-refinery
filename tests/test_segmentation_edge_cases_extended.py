# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from coreason_refinery.models import IngestionConfig
from coreason_refinery.parsing import ParsedElement
from coreason_refinery.segmentation import SemanticChunker


def test_interrupted_flow_merging() -> None:
    """
    Edge Case: "Interrupted Flow"
    Scenario: A Table followed by a "Note" (Narrative) followed by another Table,
              all within the same Header section.
    Expectation: They should ALL be merged into a single RefinedChunk because
                 there is no intervening HEADER to trigger a flush.
    """
    chunker = SemanticChunker(IngestionConfig())
    elements = [
        ParsedElement(text="# Data Section", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="| A | B |", type="TABLE", metadata={}),
        ParsedElement(text="Note: Values are approximate.", type="NARRATIVE_TEXT", metadata={}),
        ParsedElement(text="| C | D |", type="TABLE", metadata={}),
        ParsedElement(text="# Next Section", type="HEADER", metadata={"section_depth": 1}),
    ]

    chunks = chunker.chunk(elements)

    # Expected:
    # Chunk 0: Data Section (merged tables + note)
    # Chunk 1: Next Section (empty/none or depending on content)

    assert len(chunks) == 1
    content = chunks[0].text

    assert "| A | B |" in content
    assert "Note: Values are approximate." in content
    assert "| C | D |" in content
    # Order Check
    assert content.index("| A | B |") < content.index("Note:") < content.index("| C | D |")


def test_table_speaker_notes() -> None:
    """
    Edge Case: "Speaker Notes on Tables"
    Scenario: A Table element coming from a PPTX might have speaker notes attached.
    Expectation: The notes are prepended to the table content within the chunk.
    """
    chunker = SemanticChunker(IngestionConfig())
    elements = [
        ParsedElement(text="# Slide 1", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="| Q1 | Q2 |", type="TABLE", metadata={"speaker_notes": "Emphasize Q2 growth."}),
    ]

    chunks = chunker.chunk(elements)

    assert len(chunks) == 1
    content = chunks[0].text

    assert "Speaker Notes: Emphasize Q2 growth." in content
    assert "| Q1 | Q2 |" in content
    # Notes typically prepended
    assert content.index("Speaker Notes:") < content.index("| Q1 | Q2 |")


def test_hierarchy_skipping_complex() -> None:
    """
    Edge Case: "Hierarchy Skipping"
    Scenario: Jumping from H1 -> H3 -> H2.
    Expectation:
        - H1 sets context.
        - H3 pushes to stack (H1 > H3).
        - H2 should pop H3 (since 2 < 3) and append H2 (Result: H1 > H2).
    """
    chunker = SemanticChunker(IngestionConfig())
    elements = [
        ParsedElement(text="# Root", type="TITLE", metadata={}),
        # 1. H1
        ParsedElement(text="# Level 1", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Content 1", type="NARRATIVE_TEXT", metadata={}),
        # 2. H3 (Skipping H2)
        ParsedElement(text="### Level 3", type="HEADER", metadata={"section_depth": 3}),
        ParsedElement(text="Content 3", type="NARRATIVE_TEXT", metadata={}),
        # 3. H2 (Backtracking)
        ParsedElement(text="## Level 2", type="HEADER", metadata={"section_depth": 2}),
        ParsedElement(text="Content 2", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    # Expect 3 chunks (one for each content block)
    assert len(chunks) == 3

    # Chunk 0 (Content 1): Context: Root > Level 1
    assert "Content 1" in chunks[0].text
    assert "Context: Root > Level 1" in chunks[0].text

    # Chunk 1 (Content 3): Context: Root > Level 1 > Level 3
    assert "Content 3" in chunks[1].text
    assert "Context: Root > Level 1 > Level 3" in chunks[1].text

    # Chunk 2 (Content 2): Context: Root > Level 1 > Level 2
    # Logic: H2 (depth 2) is shallower than H3 (depth 3), so H3 pops.
    #        H2 (depth 2) is deeper than H1 (depth 1), so H1 stays.
    assert "Content 2" in chunks[2].text
    assert "Context: Root > Level 1 > Level 2" in chunks[2].text
