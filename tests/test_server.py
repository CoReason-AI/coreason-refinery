from typing import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_refinery.models import RefinedChunk
from coreason_refinery.server import app


# Shared mock pipeline fixture
@pytest.fixture
def mock_pipeline() -> AsyncMock:
    mock = AsyncMock()
    # Mock context manager behavior
    mock.__aenter__.return_value = mock
    mock.__aexit__.return_value = None
    return mock


@pytest.fixture
def client(mock_pipeline: AsyncMock) -> Generator[TestClient, None, None]:
    # Patch the class in server.py to return our mock
    with patch("coreason_refinery.server.RefineryPipelineAsync", return_value=mock_pipeline):
        with TestClient(app) as c:
            yield c


def test_ingest_endpoint_happy_path(client: TestClient, mock_pipeline: AsyncMock) -> None:
    """Test standard successful ingestion."""
    mock_pipeline.process.return_value = [
        RefinedChunk(id="test-chunk-id", text="Test content", vector=[0.1, 0.2], metadata={"page": 1})
    ]

    files = {"file": ("test.txt", b"dummy content", "text/plain")}

    # Mock UUID to get deterministic URN
    from uuid import UUID

    job_uuid = "12345678-1234-5678-1234-567812345678"

    with patch("coreason_refinery.server.uuid.uuid4", return_value=UUID(job_uuid)):
        response = client.post("/ingest", files=files)

    assert response.status_code == 200
    artifacts = response.json()
    assert len(artifacts) == 1

    artifact = artifacts[0]
    assert artifact["content"] == "Test content"
    assert artifact["source_location"] == {"page": 1}
    assert artifact["source_urn"] == f"urn:job:{job_uuid}:file:test.txt"
    assert artifact["artifact_type"] == "TEXT"

    assert mock_pipeline.process.called
    job = mock_pipeline.process.call_args[0][0]
    assert job.source_file_path.endswith(".txt")


def test_ingest_pipeline_failure(client: TestClient, mock_pipeline: AsyncMock) -> None:
    """Test that pipeline errors result in a 500 response."""
    mock_pipeline.process.side_effect = RuntimeError("Pipeline exploded")

    files = {"file": ("crash.pdf", b"boom", "application/pdf")}
    # FastAPI catches unhandled exceptions and returns 500
    with pytest.raises(RuntimeError):  # TestClient re-raises exceptions by default
        client.post("/ingest", files=files)


def test_ingest_empty_response(client: TestClient, mock_pipeline: AsyncMock) -> None:
    """Test when pipeline returns no chunks (e.g., empty document)."""
    mock_pipeline.process.return_value = []

    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/ingest", files=files)

    assert response.status_code == 200
    assert response.json() == []


def test_ingest_complex_artifact_mapping(client: TestClient, mock_pipeline: AsyncMock) -> None:
    """Test mapping of complex metadata and multiple chunks."""
    mock_pipeline.process.return_value = [
        RefinedChunk(id="1", text="Header", vector=[], metadata={"role": "title", "confidence": 0.99}),
        RefinedChunk(id="2", text="Cell A1", vector=[], metadata={"row": 1, "col": 1, "sheet": "Data"}),
    ]

    files = {"file": ("complex.xlsx", b"data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

    from uuid import UUID

    job_uuid = "87654321-4321-8765-4321-876543210987"

    with patch("coreason_refinery.server.uuid.uuid4", return_value=UUID(job_uuid)):
        response = client.post("/ingest", files=files)

    assert response.status_code == 200
    artifacts = response.json()
    assert len(artifacts) == 2

    # Check first artifact
    assert artifacts[0]["content"] == "Header"
    assert artifacts[0]["source_location"] == {"role": "title", "confidence": 0.99}
    assert artifacts[0]["source_urn"] == f"urn:job:{job_uuid}:file:complex.xlsx"

    # Check second artifact
    assert artifacts[1]["content"] == "Cell A1"
    assert artifacts[1]["source_location"] == {"row": 1, "col": 1, "sheet": "Data"}


def test_ingest_weird_filename(client: TestClient, mock_pipeline: AsyncMock) -> None:
    """Test handling of filenames with spaces and special characters."""
    mock_pipeline.process.return_value = []

    filename = "my cool report [final] (v2).pdf"
    files = {"file": (filename, b"content", "application/pdf")}
    response = client.post("/ingest", files=files)

    assert response.status_code == 200

    job = mock_pipeline.process.call_args[0][0]
    # Check that the temp file has the correct extension
    assert job.source_file_path.endswith(".pdf")
    # We don't enforce that the temp filename is the same, just the extension logic
