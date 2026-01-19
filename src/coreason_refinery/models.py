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


class IngestionConfig(BaseModel):  # type: ignore[misc]
    """Configuration for the ingestion process."""

    chunk_strategy: Literal["HEADER", "SIZE"] = "HEADER"
    segment_len: int = 500


class IngestionJob(BaseModel):  # type: ignore[misc]
    """Represents an ingestion job for a file."""

    id: UUID
    source_file_path: str
    file_type: str = "auto"
    config: IngestionConfig
    status: Literal["PROCESSING", "COMPLETED", "FAILED"]


class RefinedChunk(BaseModel):  # type: ignore[misc]
    """Represents a chunk of refined data ready for vector storage."""

    id: str  # Unique Vector ID
    text: str  # The Cleaned Content (Markdown)
    vector: List[float]  # Embedding
    metadata: dict[str, Any] = Field(default_factory=dict)
