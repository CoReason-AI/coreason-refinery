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
    """Chunks parsed elements based on semantic structure (headers/sections).

    Implements the 'Semantic Segmenter' (The Cutter) logic:
    1. Header-Based Splitting: Splits on Markdown headers (#, ##).
    2. Table Preservation: Merges tables spanning pages/elements.
    3. Rolling Context: Prepends header hierarchy (Breadcrumbs).
    """

    def __init__(self, config: IngestionConfig):
        self.config = config

    def _infer_depth(self, text: str) -> int:
        """Infer header depth from Markdown or numbering.

        Priorities:
        1. Markdown syntax (e.g., '##' -> 2).
        2. Labeled sections (e.g., 'Section 1.2' -> 2).
        3. Plain numbering (e.g., '1.2' -> 2).
        4. Default to 1 if no pattern matches (though typically HEADER implies some structure).
        """
        # 1. Markdown headers
        # Matches one or more '#' at the start, potentially after whitespace
        markdown_match = re.match(r"^\s*(#+)", text)
        if markdown_match:
            return len(markdown_match.group(1))

        # 2. Labeled Numbering (Section, Chapter, Part, Appendix)
        # Case insensitive. Matches "Section 1.2.3" or "Appendix A.1".
        # We allow alphanumeric numbering like "A.1" or "1.2".
        labelled_match = re.match(
            r"^\s*(?:Section|Chapter|Part|Appendix)\s+([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)",
            text,
            re.IGNORECASE,
        )
        if labelled_match:
            numbering = labelled_match.group(1)
            # Count dots + 1. "A" -> 1. "A.1" -> 2.
            return numbering.count(".") + 1

        # 3. Plain Numbering
        # Matches "1.2.3".
        match = re.match(r"^\s*(\d+(?:\.\d+)*)", text)
        if match:
            numbering = match.group(1)
            return numbering.count(".") + 1

        # Default fallback
        return 1

    def chunk(self, elements: List[ParsedElement]) -> List[RefinedChunk]:
        """Convert parsed elements into refined chunks with semantic context.

        Logic:
            - Iterate through elements.
            - Maintain `header_stack`: list of (depth, text).
            - On HEADER/TITLE:
                - Flush current buffer to a RefinedChunk.
                - Update `header_stack` based on depth.
            - On Content (Text, Table, List):
                - Append to buffer.
                - Merge metadata (e.g. page numbers).
        """
        chunks: List[RefinedChunk] = []

        # Stack entries: (depth, text)
        # TITLE is always depth 0.
        header_stack: List[Tuple[int, str]] = []

        current_buffer: List[str] = []
        current_metadata_accumulator: List[dict[str, Any]] = []

        def flush_buffer() -> None:
            """Finalize the current buffer into a RefinedChunk."""
            if not current_buffer:
                # If no content, we don't create a chunk.
                # Just clear metadata.
                current_metadata_accumulator.clear()
                return

            # 1. Join content
            # Double newline to separate paragraphs/elements cleanly in Markdown
            content_text = "\n\n".join(current_buffer)
            current_buffer.clear()

            # 2. Construct Context String (Breadcrumbs)
            # "Context: Grandparent > Parent > CurrentHeader"
            hierarchy_names = [h[1] for h in header_stack]
            if hierarchy_names:
                context_prefix = "Context: " + " > ".join(hierarchy_names)
                full_text = f"{context_prefix}\n\n{content_text}"
            else:
                full_text = content_text

            # 3. Aggregate Metadata
            aggregated_meta: dict[str, Any] = {
                "header_hierarchy": hierarchy_names,
            }

            # Collect unique page numbers
            page_numbers = set()
            for m in current_metadata_accumulator:
                if "page_number" in m:
                    page_numbers.add(m["page_number"])

            if page_numbers:
                aggregated_meta["page_numbers"] = sorted(list(page_numbers))

            # Clear accumulator
            current_metadata_accumulator.clear()

            # 4. Create Chunk
            chunks.append(
                RefinedChunk(
                    id=str(uuid.uuid4()),
                    text=full_text,
                    vector=[],  # Defer embedding
                    metadata=aggregated_meta,
                )
            )

        for element in elements:
            # -- Handle Structure (TITLE / HEADER) --
            if element.type == "TITLE":
                # A Title (like document title) resets the flow usually,
                # or sits at the very top.
                flush_buffer()
                # Depth 0. Wipe stack.
                header_stack = [(0, element.text)]

            elif element.type == "HEADER":
                flush_buffer()

                # Determine depth
                if "section_depth" in element.metadata:
                    depth = element.metadata["section_depth"]
                else:
                    depth = self._infer_depth(element.text)

                # Update Stack
                # Pop headers that are deeper or same depth (siblings/children of siblings)
                # But NEVER pop depth 0 (TITLE) if we are adding a depth >= 1.
                # Titles (0) persist until a new Title.
                while header_stack:
                    last_depth, _ = header_stack[-1]
                    # If existing header is deeper (more specific) or same depth (sibling)
                    # we pop it to make room for the new one.
                    # Exception: If existing is Root (0) and new is Header (>=1), keep Root.
                    if last_depth >= depth:
                        header_stack.pop()
                    else:
                        break

                header_stack.append((depth, element.text))

            # -- Handle Content (Text, Table, List, etc.) --
            elif element.type in [
                "NARRATIVE_TEXT",
                "TABLE",
                "LIST_ITEM",
                "UNCATEGORIZED",
            ]:
                text_to_add = element.text

                # "Speaker Notes" Extraction
                # PRD 3.1: "Must extract Speaker Notes and append them as context"
                # We prepend them to the content block.
                notes = element.metadata.get("speaker_notes")
                if notes:
                    text_to_add = f"Speaker Notes: {notes}\n{text_to_add}"

                current_buffer.append(text_to_add)

                if element.metadata:
                    current_metadata_accumulator.append(element.metadata)

            # -- Ignore Others (FOOTER, etc.) --
            # Footers are generally page artifacts, not semantic content.

        # End of elements: Flush remaining buffer
        flush_buffer()

        return chunks
