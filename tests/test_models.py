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
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from coreason_refinery.models import IngestionConfig, IngestionJob, RefinedChunk


def test_ingestion_config_defaults() -> None:
    """Test that IngestionConfig has correct defaults."""
    config = IngestionConfig()
    assert config.chunk_strategy == "HEADER"
    assert config.segment_len == 500


def test_ingestion_config_custom() -> None:
    """Test creating IngestionConfig with custom values."""
    config = IngestionConfig(chunk_strategy="SIZE", segment_len=1000)
    assert config.chunk_strategy == "SIZE"
    assert config.segment_len == 1000


def test_ingestion_config_validation() -> None:
    """Test IngestionConfig validation."""
    with pytest.raises(ValidationError):
        # Invalid chunk_strategy
        IngestionConfig(chunk_strategy="INVALID", segment_len=500)  # type: ignore[arg-type]


def test_ingestion_config_invalid_segment_len() -> None:
    """Test IngestionConfig validation for invalid segment_len."""
    # Zero
    with pytest.raises(ValidationError):
        IngestionConfig(segment_len=0)

    # Negative
    with pytest.raises(ValidationError):
        IngestionConfig(segment_len=-1)


def test_ingestion_job_creation() -> None:
    """Test creating a valid IngestionJob."""
    job_id = uuid.uuid4()
    config = IngestionConfig()
    job = IngestionJob(id=job_id, source_file_path="/tmp/test.pdf", config=config, status="PROCESSING")

    assert job.id == job_id
    assert job.source_file_path == "/tmp/test.pdf"
    assert job.config == config
    assert job.status == "PROCESSING"


def test_ingestion_job_uuid_coercion() -> None:
    """Test that IngestionJob accepts string UUIDs and coerces them."""
    job_id_str = "12345678-1234-5678-1234-567812345678"
    config = IngestionConfig()
    job = IngestionJob(
        id=job_id_str,  # type: ignore[arg-type]
        source_file_path="/tmp/test.pdf",
        config=config,
        status="PROCESSING",
    )
    assert isinstance(job.id, uuid.UUID)
    assert str(job.id) == job_id_str


def test_ingestion_job_dict_config() -> None:
    """Test that IngestionJob accepts dict for config and coerces it."""
    job_id = uuid.uuid4()
    config_dict = {"chunk_strategy": "SIZE", "segment_len": 200}
    job = IngestionJob(
        id=job_id,
        source_file_path="/tmp/test.pdf",
        config=config_dict,  # type: ignore[arg-type]
        status="PROCESSING",
    )
    assert isinstance(job.config, IngestionConfig)
    assert job.config.chunk_strategy == "SIZE"
    assert job.config.segment_len == 200


def test_ingestion_job_validation() -> None:
    """Test IngestionJob validation."""
    job_id = uuid.uuid4()
    config = IngestionConfig()

    with pytest.raises(ValidationError):
        # Invalid status
        IngestionJob(
            id=job_id,
            source_file_path="/tmp/test.pdf",
            config=config,
            status="INVALID",  # type: ignore[arg-type]
        )


def test_refined_chunk_creation() -> None:
    """Test creating a valid RefinedChunk."""
    metadata: Dict[str, Any] = {"source_urn": "mcp://sharepoint/doc.pdf", "page_num": 14, "is_table": True}
    chunk = RefinedChunk(id="vector_id_123", text="# Title\n\nContent", vector=[0.1, 0.2, 0.3], metadata=metadata)

    assert chunk.id == "vector_id_123"
    assert chunk.text == "# Title\n\nContent"
    assert chunk.vector == [0.1, 0.2, 0.3]
    assert chunk.metadata == metadata


def test_refined_chunk_defaults() -> None:
    """Test RefinedChunk defaults."""
    chunk = RefinedChunk(id="vector_id_123", text="Content", vector=[0.1])
    assert chunk.metadata == {}


def test_refined_chunk_complex_metadata() -> None:
    """Test RefinedChunk with complex metadata structures."""
    metadata: Dict[str, Any] = {
        "simple": "value",
        "nested": {"key": "value", "list": [1, 2, 3]},
        "list_of_dicts": [{"a": 1}, {"b": 2}],
        "null_value": None,
    }
    chunk = RefinedChunk(id="vector_id_complex", text="Content", vector=[0.1], metadata=metadata)
    assert chunk.metadata == metadata
    assert chunk.metadata["nested"]["list"][1] == 2


def test_refined_chunk_vector_validation() -> None:
    """Test RefinedChunk vector validation (although List[float] allows ints, it should fail strings)."""
    with pytest.raises(ValidationError):
        RefinedChunk(
            id="vector_id",
            text="Content",
            vector=["not", "a", "float"],  # type: ignore[list-item]
        )
