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

    # Chunk 0: Title > Section 1
    # Note: "Title" (ParsedElement type TITLE) is depth 0.
    # "# Section 1" should be depth 1.
    assert chunks[0].metadata["header_hierarchy"] == ["# Title", "# Section 1"]

    # Chunk 1: Title > Section 1 > Subsection 1.1
    # "## Subsection 1.1" should be depth 2.
    assert chunks[1].metadata["header_hierarchy"] == ["# Title", "# Section 1", "## Subsection 1.1"]

    # Chunk 2: Title > Section 1 > Subsection 1.1 > Detail 1.1.1
    # "### Detail 1.1.1" should be depth 3.
    assert chunks[2].metadata["header_hierarchy"] == ["# Title", "# Section 1", "## Subsection 1.1", "### Detail 1.1.1"]

    # Chunk 3: Title > Section 2
    # "# Section 2" (Depth 1) should pop Depth 1, 2, 3.
    # Stack: Title (0) -> Section 2 (1).
    assert chunks[3].metadata["header_hierarchy"] == ["# Title", "# Section 2"]
