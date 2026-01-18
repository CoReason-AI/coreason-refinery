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


def test_story_a_table_rescue_large_table() -> None:
    """
    Story A: The "Table Rescue" (Structure Preservation)
    Context: User uploads a PDF with a multi-page dosing table.
    Requirement: A Table must never be split mid-row. If it spans pages, it is merged into one logical block.
    Constraint Check: Ensure table is NOT split even if it exceeds segment_len (e.g. 500 chars).
    """
    # Config with a small segment length to trigger potential splitting if it were implemented naively
    config = IngestionConfig(chunk_strategy="HEADER", segment_len=100)
    chunker = SemanticChunker(config)

    # Create a large table split across multiple elements (simulating pages)
    # Total length > 100 chars
    row_part_1 = "| Dose | Response |\n| --- | --- |\n" + ("| 10mg | Good |\n" * 5)  # ~80 chars
    row_part_2 = "| 20mg | Better |\n" * 5  # ~80 chars
    row_part_3 = "| 30mg | Best |\n" * 5  # ~80 chars

    elements = [
        ParsedElement(text="# Protocol 999", type="TITLE", metadata={"page_number": 1}),
        ParsedElement(text="## Section 4: Toxicity", type="HEADER", metadata={"page_number": 1, "section_depth": 2}),
        # Table starts on page 1
        ParsedElement(text=row_part_1, type="TABLE", metadata={"page_number": 1, "is_table": True}),
        # Table continues on page 2 (no intervening header)
        ParsedElement(text=row_part_2, type="TABLE", metadata={"page_number": 2, "is_table": True}),
        # Table continues on page 3
        ParsedElement(text=row_part_3, type="TABLE", metadata={"page_number": 3, "is_table": True}),
        # Next section
        ParsedElement(text="## Section 5: Conclusion", type="HEADER", metadata={"page_number": 3, "section_depth": 2}),
        # Add content to section 5 so it becomes a chunk
        ParsedElement(text="The study is concluded.", type="NARRATIVE_TEXT", metadata={"page_number": 3}),
    ]

    chunks = chunker.chunk(elements)

    # Expectations:
    # Chunk 0: Protocol 999 (Title) -> No content, so NO chunk.
    # Chunk 1: Section 4 + Merged Table
    # Chunk 2: Section 5 + Narrative

    # Note: If the code were to split the table due to size, we would see more chunks here.
    # The fact that we only get 2 chunks (one for table, one for conclusion) proves atomicity.

    assert len(chunks) == 2

    # Check Chunk 0 (The Table)
    table_chunk = chunks[0]

    # Verify Content Merging
    assert row_part_1 in table_chunk.text
    assert row_part_2 in table_chunk.text
    assert row_part_3 in table_chunk.text

    # Verify Context (Story B check implicitly)
    assert "Context: # Protocol 999 > ## Section 4: Toxicity" in table_chunk.text

    # Verify Metadata Aggregation (Page numbers)
    assert "page_numbers" in table_chunk.metadata
    assert table_chunk.metadata["page_numbers"] == [1, 2, 3]


def test_story_b_header_context_breadcrumbs() -> None:
    """
    Story B: The "Header Context" (Semantic Context)
    Context: A chunk simply says: "Stop treatment immediately."
    Action: Refinery prepends the hierarchy: Context: [Protocol 999] > [Section 4: Toxicity] > [Grade 4 Events].
    """
    config = IngestionConfig()
    chunker = SemanticChunker(config)

    elements = [
        ParsedElement(text="Protocol 999", type="TITLE", metadata={}),
        ParsedElement(text="Section 4: Toxicity", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text="Grade 4 Events", type="HEADER", metadata={"section_depth": 2}),
        ParsedElement(text="Stop treatment immediately.", type="NARRATIVE_TEXT", metadata={}),
    ]

    chunks = chunker.chunk(elements)

    # Last chunk should contain the text and the context
    target_chunk = chunks[-1]

    assert "Stop treatment immediately." in target_chunk.text
    # Check strict format: "Context: Root > Section \n\n [Content]"
    expected_context = "Context: Protocol 999 > Section 4: Toxicity > Grade 4 Events"
    assert expected_context in target_chunk.text
    assert target_chunk.text.startswith(expected_context)


def test_single_large_table_atomicity() -> None:
    """
    AUC-5: Verify single large table element atomicity.
    Requirement: A single TABLE element exceeding segment_len must NOT be split.
    """
    # Set segment_len to be very small
    segment_len = 50
    config = IngestionConfig(chunk_strategy="HEADER", segment_len=segment_len)
    chunker = SemanticChunker(config)

    # Create a large table text significantly larger than segment_len
    large_table_text = "| Col1 | Col2 |\n| --- | --- |\n"
    for i in range(20):
        large_table_text += f"| Row {i} | Data {i} |\n"

    # Verify we created a large enough string
    assert len(large_table_text) > segment_len * 2

    elements = [
        ParsedElement(text="# Header", type="HEADER", metadata={"section_depth": 1}),
        ParsedElement(text=large_table_text, type="TABLE", metadata={"page_number": 1}),
    ]

    chunks = chunker.chunk(elements)

    # Should be exactly 1 chunk (Header + Table)
    assert len(chunks) == 1

    chunk = chunks[0]
    # Verify the entire table text is present in the chunk
    assert large_table_text in chunk.text
    # Verify length is > segment_len (plus context overhead)
    assert len(chunk.text) > segment_len
