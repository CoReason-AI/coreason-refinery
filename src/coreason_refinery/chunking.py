# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

import uuid
from typing import List, Tuple

from coreason_refinery.models import IngestionConfig, RefinedChunk
from coreason_refinery.parsing import ParsedElement


class SemanticChunker:
    """Chunks parsed elements based on semantic structure (headers/sections)."""

    def __init__(self, config: IngestionConfig):
        self.config = config

    def chunk(self, elements: List[ParsedElement]) -> List[RefinedChunk]:
        """Convert parsed elements into refined chunks with semantic context.

        Strategy:
            - Iterate through elements.
            - Maintain a stack of active headers (context).
            - When a HEADER/TITLE is found:
                1. Flush the current text buffer as a new chunk (if not empty).
                2. Update the header stack based on depth.
            - When CONTENT (Text, Table, List) is found:
                1. Append to current text buffer.
        """
        chunks: List[RefinedChunk] = []
        # Stack of (depth, text).
        # TITLE is depth 0. HEADER defaults to 1 unless specified.
        header_stack: List[Tuple[int, str]] = []

        current_buffer: List[str] = []

        # Helper to flush buffer
        def flush_buffer() -> None:
            if not current_buffer:
                return

            # Join content
            content_text = "\n\n".join(current_buffer)
            current_buffer.clear()

            # Construct Context String
            # "Context: Grandparent > Parent > Section \n\n [Actual Content]"
            hierarchy_names = [h[1] for h in header_stack]
            if hierarchy_names:
                context_prefix = "Context: " + " > ".join(hierarchy_names)
                full_text = f"{context_prefix}\n\n{content_text}"
            else:
                full_text = content_text

            # Create Chunk
            chunks.append(
                RefinedChunk(
                    id=str(uuid.uuid4()),
                    text=full_text,
                    vector=[],  # To be computed later
                    metadata={
                        "header_hierarchy": hierarchy_names,
                        # We could add source_urn etc if available in element metadata,
                        # but usually that comes from the Job level.
                        # We'll merge element metadata later if needed.
                    },
                )
            )

        for element in elements:
            if element.type == "TITLE":
                # Flush previous section
                flush_buffer()

                # Title is always root (0). Clear stack and add Title.
                header_stack = [(0, element.text)]

            elif element.type == "HEADER":
                # Flush previous section
                flush_buffer()

                # Determine depth
                # Default to 1 if not specified.
                # If we want to be smart, we could look at font size in metadata,
                # but for now we rely on 'section_depth' or default.
                depth = element.metadata.get("section_depth", 1)

                # Pop headers that are deeper or equal to this new header
                # Example: Stack=[(0, Title), (1, Intro)]. New is (1, Methods).
                # Pop (1, Intro). Stack=[(0, Title), (1, Methods)].
                while header_stack and header_stack[-1][0] >= depth:
                    header_stack.pop()

                header_stack.append((depth, element.text))

            elif element.type in [
                "NARRATIVE_TEXT",
                "TABLE",
                "LIST_ITEM",
                "UNCATEGORIZED",
            ]:
                # Append to buffer
                current_buffer.append(element.text)

            # Handle footer? Usually ignore or treat as metadata.
            # PRD mentions "Headers, Footers" in "Structure-Aware".
            # Parsing.py has FOOTER type.
            elif element.type == "FOOTER":
                # We generally ignore footers in the main narrative text
                # to avoid breaking the flow, or add them as metadata.
                # For now, let's ignore them in the text stream.
                pass

        # Flush any remaining buffer at the end
        flush_buffer()

        return chunks
