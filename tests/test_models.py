from uuid import uuid4

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
