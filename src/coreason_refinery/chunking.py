# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

import re
import uuid
from typing import Any, List, Tuple

from coreason_refinery.models import IngestionConfig, RefinedChunk
from coreason_refinery.parsing import ParsedElement


class SemanticChunker:
    """Chunks parsed elements based on semantic structure (headers/sections)."""

    def __init__(self, config: IngestionConfig):
        self.config = config

    def _infer_depth(self, text: str) -> int:
        """Infer header depth from numbering (e.g., '1.2' -> 2) or markdown (e.g., '##' -> 2)."""
        # Check Markdown headers first
        markdown_match = re.match(r"^\s*(#+)", text)
        if markdown_match:
            return len(markdown_match.group(1))

        # Check for labelled numbering like "Section 1.2" or "Chapter 3.1.1"
        # We look for (Word) (Numbering)
        labelled_match = re.match(r"^\s*(?:Section|Chapter|Part|Appendix)\s+(\d+(\.\d+)*)", text, re.IGNORECASE)
        if labelled_match:
            numbering = labelled_match.group(1)
            return numbering.count(".") + 1

        # Fallback to plain numbering
        match = re.match(r"^\s*(\d+(\.\d+)*)", text)
        if match:
            # "1" -> 1 (0 dots)
            # "1.1" -> 2 (1 dot)
            return match.group(1).count(".") + 1
        return 1

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
        # Accumulate metadata from elements in the current chunk
        current_metadata_accumulator: List[dict[str, Any]] = []

        # Helper to flush buffer
        def flush_buffer() -> None:
            if not current_buffer:
                # Also clear metadata accumulator if we are not creating a chunk
                current_metadata_accumulator.clear()
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

            # Merge metadata
            aggregated_meta: dict[str, Any] = {
                "header_hierarchy": hierarchy_names,
            }

            # Collect page numbers specifically
            page_numbers = set()
            for m in current_metadata_accumulator:
                if "page_number" in m:
                    page_numbers.add(m["page_number"])
                # We could merge other keys here if needed

            if page_numbers:
                aggregated_meta["page_numbers"] = sorted(list(page_numbers))

            current_metadata_accumulator.clear()

            # Create Chunk
            chunks.append(
                RefinedChunk(
                    id=str(uuid.uuid4()),
                    text=full_text,
                    vector=[],  # To be computed later
                    metadata=aggregated_meta,
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
                if "section_depth" in element.metadata:
                    depth = element.metadata["section_depth"]
                else:
                    depth = self._infer_depth(element.text)

                # Pop headers that are deeper or equal to this new header.
                # Crucial Fix: We must NOT pop depth 0 (TITLE) if the new depth is >= 1.
                # Titles are roots (depth 0). Headers are depth >= 1.
                # If we encounter a Header with depth 1, we should only pop existing headers
                # that are >= 1. We keep 0.
                while header_stack and header_stack[-1][0] >= depth:
                    header_stack.pop()

                header_stack.append((depth, element.text))

            elif element.type in [
                "NARRATIVE_TEXT",
                "TABLE",
                "LIST_ITEM",
                "UNCATEGORIZED",
            ]:
                text_to_add = element.text

                # Handle speaker notes
                notes = element.metadata.get("speaker_notes")
                if notes:
                    # Prepend speaker notes as context for this element
                    text_to_add = f"Speaker Notes: {notes}\n{text_to_add}"

                # Append to buffer
                current_buffer.append(text_to_add)

                # Accumulate metadata
                if element.metadata:
                    current_metadata_accumulator.append(element.metadata)

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
