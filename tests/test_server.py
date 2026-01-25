# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_refinery.models import RefinedChunk
from coreason_refinery.server import app


@pytest.fixture
def mock_pipeline() -> Generator[MagicMock, None, None]:
    with patch("coreason_refinery.server.RefineryPipelineAsync") as MockPipeline:
        pipeline_instance = MockPipeline.return_value
        pipeline_instance.__aenter__ = AsyncMock(return_value=pipeline_instance)
        pipeline_instance.__aexit__ = AsyncMock(return_value=None)
        pipeline_instance.process = AsyncMock(return_value=[RefinedChunk(id="1", text="test", vector=[], metadata={})])
        yield pipeline_instance


def test_ingest_endpoint(mock_pipeline: MagicMock) -> None:
    with TestClient(app) as client:
        response = client.post("/ingest", files={"file": ("test.txt", b"content", "text/plain")})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["text"] == "test"

        # Verify process was called
        assert mock_pipeline.process.called


def test_ingest_no_file() -> None:
    with TestClient(app) as client:
        response = client.post("/ingest")
        assert response.status_code == 422


def test_ingest_error(mock_pipeline: MagicMock) -> None:
    mock_pipeline.process.side_effect = RuntimeError("Processing failed")
    with TestClient(app) as client:
        # FastAPI TestClient re-raises exceptions from the app
        with pytest.raises(RuntimeError, match="Processing failed"):
            client.post("/ingest", files={"file": ("test.txt", b"content", "text/plain")})


def test_ingest_cleanup_error(mock_pipeline: MagicMock) -> None:
    # We need to simulate the file existing so cleanup tries to remove it
    # Then make remove fail
    with TestClient(app) as client:
        # We need to patch os.remove to raise exception
        # We assume tempfile creation works fine
        with patch("coreason_refinery.server.os.remove", side_effect=OSError("Permission denied")):
            response = client.post("/ingest", files={"file": ("test.txt", b"content", "text/plain")})
            assert response.status_code == 200
