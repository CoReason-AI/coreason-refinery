# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from typing import Any, List, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration for the ingestion process.

    Attributes:
        chunk_strategy: The strategy to use for chunking.
            Defaults to "HEADER" (semantic segmentation).
            "SIZE" is reserved for future use or legacy fallback.
        segment_len: The target length for a segment/chunk in characters.
            The semantic segmenter tries to respect this but prioritizes
            document structure (e.g. keeping tables intact).
    """

    chunk_strategy: Literal["HEADER", "SIZE"] = "HEADER"
    segment_len: int = 500


class IngestionJob(BaseModel):
    """Represents an ingestion job for a file.

    Attributes:
        id: Unique identifier for the job.
        source_file_path: The URI or file path of the source document.
        file_type: The type of the source file (e.g., 'pdf', 'xlsx', 'pptx').
            Defaults to 'auto' for extension-based inference.
        config: Configuration parameters for this ingestion job.
        status: The current status of the job.
    """

    id: UUID
    source_file_path: str
    file_type: str = "auto"
    config: IngestionConfig
    status: Literal["PROCESSING", "COMPLETED", "FAILED"]


class RefinedChunk(BaseModel):
    """Represents a chunk of refined data ready for vector storage.

    Corresponds to 'RefinedSegment' in the PRD.

    Attributes:
        id: Unique Vector ID for this chunk.
        text: The Cleaned Content, formatted as Markdown.
            Contains context-enriched text (e.g. including headers).
        vector: The embedding vector for the text (deferred generation).
        metadata: Enriched metadata for the chunk.
            Includes:
            - source_urn (from job)
            - header_hierarchy (The Rolling Context)
            - page_numbers
            - contains_table / contains_equation
            - speaker_notes_included (for PPTX)
    """

    id: str
    text: str
    vector: List[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
