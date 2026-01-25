from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from coreason_refinery.models import RefinedChunk
from coreason_refinery.server import app


def test_ingest_endpoint() -> None:
    # Mock the pipeline instance
    mock_pipeline = AsyncMock()
    mock_pipeline.process.return_value = [
        RefinedChunk(id="test-chunk-id", text="Test content", vector=[0.1, 0.2], metadata={"page": 1})
    ]

    # Mock the context manager behavior of the pipeline
    mock_pipeline.__aenter__.return_value = mock_pipeline
    mock_pipeline.__aexit__.return_value = None

    # Patch the class in server.py to return our mock
    # We patch where it is used (in server.py lifespan)
    with patch("coreason_refinery.server.RefineryPipelineAsync", return_value=mock_pipeline):
        with TestClient(app) as client:
            files = {"file": ("test.txt", b"dummy content", "text/plain")}
            response = client.post("/ingest", files=files)

            assert response.status_code == 200
            artifacts = response.json()
            assert len(artifacts) == 1

            artifact = artifacts[0]
            assert artifact["content"] == "Test content"
            assert artifact["source_location"] == {"page": 1}
            assert artifact["source_urn"] == "urn:file:test.txt"
            assert artifact["artifact_type"] == "TEXT"

            # Verify pipeline process was called
            assert mock_pipeline.process.called
            call_args = mock_pipeline.process.call_args
            job = call_args[0][0]
            assert job.status == "PROCESSING"
            assert job.source_file_path.endswith(".txt")
