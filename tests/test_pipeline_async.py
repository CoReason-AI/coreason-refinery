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
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from coreason_refinery.models import IngestionConfig, IngestionJob, RefinedChunk
from coreason_refinery.parsing import ParsedElement
from coreason_refinery.pipeline import RefineryPipelineAsync


@pytest.fixture
def sample_job() -> IngestionJob:
    return IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/tmp/test.pdf",
        config=IngestionConfig(),
        status="PROCESSING",
        file_type="pdf",
    )


@pytest.fixture
def mock_elements() -> List[ParsedElement]:
    return [
        ParsedElement(text="Title", type="TITLE", metadata={}),
    ]


@pytest.mark.asyncio
async def test_pipeline_async_process(sample_job: IngestionJob, mock_elements: List[ParsedElement]) -> None:
    """Test the async process method."""
    with (
        patch("coreason_refinery.pipeline.UnstructuredPdfParser") as MockPdfParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        # Setup Mocks
        mock_parser = MockPdfParser.return_value
        mock_parser.parse.return_value = mock_elements

        mock_chunker = MockChunker.return_value
        mock_chunker.chunk.return_value = [RefinedChunk(id="1", text="Chunk 1", vector=[])]

        async with RefineryPipelineAsync() as pipeline:
            result = await pipeline.process(sample_job)

            assert len(result) == 1
            assert result[0].text == "Chunk 1"

            # Verify to_thread usage indirectly by checking methods were called
            mock_parser.parse.assert_called_once_with("/tmp/test.pdf")
            mock_chunker.chunk.assert_called_once_with(mock_elements)


@pytest.mark.asyncio
async def test_resource_cleanup() -> None:
    """Test that the internal client is closed on exit."""
    # We mock httpx.AsyncClient to verify aclose is called
    with patch("httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        MockClient.return_value = mock_client_instance

        async with RefineryPipelineAsync() as pipeline:
            assert pipeline._internal_client is True
            # Do something
            pass

        mock_client_instance.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_external_client_lifecycle() -> None:
    """Test that an external client is NOT closed."""
    mock_external_client = AsyncMock()

    async with RefineryPipelineAsync(client=mock_external_client) as pipeline:
        assert pipeline._internal_client is False
        pass

    mock_external_client.aclose.assert_not_called()
