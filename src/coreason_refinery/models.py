# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from typing import Any, Dict, List, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """
    Configuration for the ingestion process.

    Attributes:
        use_ocr (bool): Whether to force OCR for PDF processing.
        split_strategy (str): The strategy used for splitting documents (e.g., 'MARKDOWN_HEADER').
    """

    use_ocr: bool = False
    split_strategy: str = "MARKDOWN_HEADER"


class RefinedChunk(BaseModel):
    """
    A semantic chunk of a processed document, enriched with metadata and embeddings.

    Attributes:
        id (str): Unique identifier for the chunk.
        text_markdown (str): The cleaned, markdown-formatted content.
        vector (List[float]): The semantic embedding vector.
        metadata (dict): Contextual metadata (hierarchy, source, table presence, etc.).
    """

    id: str
    text_markdown: str
    vector: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionJob(BaseModel):
    """
    Represents a job to ingest a specific document source.

    Attributes:
        id (UUID): Unique job identifier.
        source_file_path (str): Path or URN to the source file.
        file_type (str): Type of file ('pdf', 'xlsx', 'pptx', 'auto').
        config (IngestionConfig): Configuration parameters for the job.
        status (Literal): Current status of the job.
    """

    id: UUID
    source_file_path: str
    file_type: str = "auto"
    config: IngestionConfig
    status: Literal["PROCESSING", "COMPLETED", "FAILED"]


class ParsedElement(BaseModel):
    """
    An atomic element parsed from a raw document.

    Attributes:
        type (Literal): The type of the element (e.g., 'TITLE', 'TABLE').
        text (str): The raw text content of the element.
        metadata (dict): Metadata associated with the element (page number, coordinates, etc.).
    """

    type: Literal["TITLE", "NARRATIVE_TEXT", "TABLE", "LIST_ITEM", "HEADER", "FOOTER", "UNCATEGORIZED"]
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
