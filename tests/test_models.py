# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from typing import Any, Dict
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from coreason_refinery.models import IngestionConfig, IngestionJob, RefinedChunk


def test_ingestion_config_defaults() -> None:
    """Test IngestionConfig defaults."""
    config = IngestionConfig()
    assert config.use_ocr is False
    assert config.split_strategy == "MARKDOWN_HEADER"


def test_refined_chunk_creation() -> None:
    """Test RefinedChunk creation and defaults."""
    chunk = RefinedChunk(id="test-id", text_markdown="# Header\nContent")
    assert chunk.id == "test-id"
    assert chunk.text_markdown == "# Header\nContent"
    assert chunk.vector == []
    assert chunk.metadata == {}


def test_refined_chunk_with_data() -> None:
    """Test RefinedChunk with explicit vector and metadata."""
    vector = [0.1, 0.2, 0.3]
    metadata = {"source": "test.pdf"}
    chunk = RefinedChunk(id="test-id", text_markdown="content", vector=vector, metadata=metadata)
    assert chunk.vector == vector
    assert chunk.metadata == metadata


def test_ingestion_job_creation() -> None:
    """Test IngestionJob creation."""
    job_id = uuid4()
    config = IngestionConfig(use_ocr=True)
    job = IngestionJob(
        id=job_id,
        source_file_path="/path/to/doc.pdf",
        config=config,
        status="PROCESSING",
    )
    assert job.id == job_id
    assert job.source_file_path == "/path/to/doc.pdf"
    assert job.file_type == "auto"
    assert job.config.use_ocr is True
    assert job.status == "PROCESSING"


def test_ingestion_job_validation_error() -> None:
    """Test IngestionJob validation error for invalid status."""
    job_id = uuid4()
    config = IngestionConfig()
    with pytest.raises(ValidationError):
        IngestionJob(
            id=job_id,
            source_file_path="doc.pdf",
            config=config,
            status="INVALID_STATUS",  # type: ignore[arg-type]
        )


def test_ingestion_job_explicit_file_type() -> None:
    """Test IngestionJob with explicit file type."""
    job_id = uuid4()
    config = IngestionConfig()
    job = IngestionJob(
        id=job_id,
        source_file_path="doc.xlsx",
        file_type="xlsx",
        config=config,
        status="COMPLETED",
    )
    assert job.file_type == "xlsx"


# --- Extended Edge Cases & Complex Scenarios ---


def test_refined_chunk_complex_metadata() -> None:
    """Test RefinedChunk with complex nested metadata."""
    complex_metadata: Dict[str, Any] = {
        "source": "test.pdf",
        "hierarchy": ["Root", "Section 1", "Subsection A"],
        "stats": {"word_count": 100, "confidence": 0.95},
        "tags": ["urgent", "confidential"],
        "is_valid": True,
    }
    chunk = RefinedChunk(
        id="complex-id",
        text_markdown="Complex Content",
        metadata=complex_metadata,
    )
    assert chunk.metadata == complex_metadata
    assert chunk.metadata["hierarchy"][1] == "Section 1"
    assert chunk.metadata["stats"]["word_count"] == 100


def test_ingestion_job_uuid_coercion() -> None:
    """Test that IngestionJob accepts a string UUID and coerces it to a UUID object."""
    uuid_str = "12345678-1234-5678-1234-567812345678"
    config = IngestionConfig()
    job = IngestionJob(
        id=uuid_str,  # type: ignore[arg-type]
        source_file_path="doc.pdf",
        config=config,
        status="PROCESSING",
    )
    assert isinstance(job.id, UUID)
    assert str(job.id) == uuid_str


def test_ingestion_job_invalid_uuid() -> None:
    """Test that IngestionJob raises ValidationError for an invalid UUID string."""
    config = IngestionConfig()
    with pytest.raises(ValidationError):
        IngestionJob(
            id="not-a-uuid",  # type: ignore[arg-type]
            source_file_path="doc.pdf",
            config=config,
            status="PROCESSING",
        )


def test_ingestion_job_json_roundtrip() -> None:
    """Test JSON serialization and deserialization (round-trip) for IngestionJob."""
    job_id = uuid4()
    config = IngestionConfig(use_ocr=True, split_strategy="CUSTOM_STRATEGY")
    original_job = IngestionJob(
        id=job_id,
        source_file_path="mcp://sharepoint/doc.pdf",
        file_type="pdf",
        config=config,
        status="COMPLETED",
    )

    # Serialize to JSON
    json_str = original_job.model_dump_json()

    # Validate back to object
    restored_job = IngestionJob.model_validate_json(json_str)

    assert restored_job == original_job
    assert restored_job.id == original_job.id
    assert restored_job.config.use_ocr == original_job.config.use_ocr


def test_refined_chunk_json_roundtrip() -> None:
    """Test JSON serialization and deserialization (round-trip) for RefinedChunk."""
    chunk = RefinedChunk(
        id="chunk-1",
        text_markdown="# Title",
        vector=[0.1, 0.9],
        metadata={"key": "value"},
    )

    json_str = chunk.model_dump_json()
    restored_chunk = RefinedChunk.model_validate_json(json_str)

    assert restored_chunk == chunk
    assert restored_chunk.vector == chunk.vector
