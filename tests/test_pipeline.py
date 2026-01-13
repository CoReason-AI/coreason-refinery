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
from unittest.mock import patch

import pytest

from coreason_refinery.models import IngestionConfig, IngestionJob, RefinedChunk
from coreason_refinery.parsing import ParsedElement
from coreason_refinery.pipeline import RefineryPipeline


@pytest.fixture
def pipeline() -> RefineryPipeline:
    return RefineryPipeline()


@pytest.fixture
def sample_job() -> IngestionJob:
    return IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/tmp/test.pdf",
        config=IngestionConfig(),
        status="PROCESSING",
    )


@pytest.fixture
def mock_elements() -> List[ParsedElement]:
    return [
        ParsedElement(text="Title", type="TITLE", metadata={}),
        ParsedElement(text="Header", type="HEADER", metadata={}),
        ParsedElement(text="Content", type="NARRATIVE_TEXT", metadata={}),
    ]


def test_process_pdf_auto_detection(
    pipeline: RefineryPipeline,
    sample_job: IngestionJob,
    mock_elements: List[ParsedElement],
) -> None:
    """Test that PDF files are correctly routed to UnstructuredPdfParser."""
    # Mock the parser and chunker
    with (
        patch("coreason_refinery.pipeline.UnstructuredPdfParser") as MockPdfParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        # Setup mocks
        mock_parser_instance = MockPdfParser.return_value
        mock_parser_instance.parse.return_value = mock_elements

        mock_chunker_instance = MockChunker.return_value
        mock_chunker_instance.chunk.return_value = [RefinedChunk(id="1", text="Chunk 1", vector=[])]

        # Ensure job is set to auto
        sample_job.file_type = "auto"
        sample_job.source_file_path = "/tmp/doc.pdf"

        # Execute
        result = pipeline.process(sample_job)

        # Verify
        MockPdfParser.assert_called_once()
        mock_parser_instance.parse.assert_called_once_with("/tmp/doc.pdf")
        MockChunker.assert_called_once_with(sample_job.config)
        mock_chunker_instance.chunk.assert_called_once_with(mock_elements)
        assert len(result) == 1
        assert result[0].text == "Chunk 1"


def test_process_excel_explicit_type(
    pipeline: RefineryPipeline,
    sample_job: IngestionJob,
    mock_elements: List[ParsedElement],
) -> None:
    """Test explicit Excel file type routing."""
    with (
        patch("coreason_refinery.pipeline.ExcelParser") as MockExcelParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        mock_parser_instance = MockExcelParser.return_value
        mock_parser_instance.parse.return_value = mock_elements

        mock_chunker_instance = MockChunker.return_value
        mock_chunker_instance.chunk.return_value = []

        sample_job.file_type = "excel"
        sample_job.source_file_path = "/tmp/data.xlsx"

        pipeline.process(sample_job)

        MockExcelParser.assert_called_once()


def test_unsupported_file_type(pipeline: RefineryPipeline) -> None:
    """Test handling of unsupported file types."""
    job = IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/tmp/unknown.xyz",
        file_type="xyz",
        config=IngestionConfig(),
        status="PROCESSING",
    )

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process(job)

    assert "Unsupported file type: xyz" in str(excinfo.value)


def test_pipeline_failure_handling(pipeline: RefineryPipeline, sample_job: IngestionJob) -> None:
    """Test that pipeline wraps exceptions in RuntimeError."""
    with patch("coreason_refinery.pipeline.UnstructuredPdfParser") as MockPdfParser:
        mock_parser_instance = MockPdfParser.return_value
        mock_parser_instance.parse.side_effect = Exception("Parsing crashed")

        sample_job.file_type = "pdf"

        with pytest.raises(RuntimeError) as excinfo:
            pipeline.process(sample_job)

        assert "Pipeline processing failed: Parsing crashed" in str(excinfo.value)


def test_auto_detection_xls(pipeline: RefineryPipeline, mock_elements: List[ParsedElement]) -> None:
    """Test auto-detection for .xls files."""
    job = IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/tmp/legacy.xls",
        file_type="auto",
        config=IngestionConfig(),
        status="PROCESSING",
    )

    with (
        patch("coreason_refinery.pipeline.ExcelParser") as MockExcelParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        MockExcelParser.return_value.parse.return_value = mock_elements
        MockChunker.return_value.chunk.return_value = []

        pipeline.process(job)

        MockExcelParser.assert_called_once()


def test_auto_detection_csv(pipeline: RefineryPipeline, mock_elements: List[ParsedElement]) -> None:
    """Test auto-detection for .csv files (mapped to Excel parser)."""
    job = IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/tmp/data.csv",
        file_type="auto",
        config=IngestionConfig(),
        status="PROCESSING",
    )

    with (
        patch("coreason_refinery.pipeline.ExcelParser") as MockExcelParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        MockExcelParser.return_value.parse.return_value = mock_elements
        MockChunker.return_value.chunk.return_value = []

        pipeline.process(job)

        MockExcelParser.assert_called_once()


def test_auto_detection_failure(pipeline: RefineryPipeline) -> None:
    """Test auto-detection failure for unknown extensions."""
    job = IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/tmp/unknown.unknown",
        file_type="auto",
        config=IngestionConfig(),
        status="PROCESSING",
    )

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process(job)

    # It wraps ValueError in RuntimeError
    assert "Unsupported file type: auto" in str(excinfo.value)


# --- New Edge Case Tests ---


def test_process_empty_document(pipeline: RefineryPipeline, sample_job: IngestionJob) -> None:
    """Test processing a document that results in zero parsed elements."""
    with (
        patch("coreason_refinery.pipeline.UnstructuredPdfParser") as MockPdfParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        # Parser returns empty list
        MockPdfParser.return_value.parse.return_value = []
        # Chunker returns empty list
        MockChunker.return_value.chunk.return_value = []

        sample_job.file_type = "pdf"
        result = pipeline.process(sample_job)

        assert result == []
        MockPdfParser.return_value.parse.assert_called_once()
        MockChunker.return_value.chunk.assert_called_once_with([])


def test_process_case_insensitive_extension(pipeline: RefineryPipeline) -> None:
    """Test that file extensions are case-insensitive (e.g., .PDF)."""
    job = IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/docs/IMPORTANT.PDF",
        file_type="auto",
        config=IngestionConfig(),
        status="PROCESSING",
    )

    with (
        patch("coreason_refinery.pipeline.UnstructuredPdfParser") as MockPdfParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        MockPdfParser.return_value.parse.return_value = []
        MockChunker.return_value.chunk.return_value = []

        pipeline.process(job)

        # Should select PDF parser
        MockPdfParser.assert_called_once()


def test_process_no_extension(pipeline: RefineryPipeline) -> None:
    """Test behavior when file has no extension."""
    job = IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/bin/executable_file",
        file_type="auto",
        config=IngestionConfig(),
        status="PROCESSING",
    )

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process(job)

    # "Unsupported file type: auto" because no extension match found
    assert "Unsupported file type: auto" in str(excinfo.value)


def test_complex_data_flow(pipeline: RefineryPipeline) -> None:
    """Test a full data flow scenario: Job -> Parser -> Chunker -> Result.

    Verifies that configuration is passed to the chunker and the result from
    the chunker is returned by the pipeline.
    """
    # 1. Setup Job
    job = IngestionJob(
        id=uuid.uuid4(),
        source_file_path="/tmp/complex.pdf",
        file_type="pdf",
        config=IngestionConfig(chunk_strategy="HEADER", segment_len=1000),
        status="PROCESSING",
    )

    # 2. Mock Data
    elements = [ParsedElement(text="H1", type="HEADER", metadata={"depth": 1})]
    chunks = [RefinedChunk(id="c1", text="Context: H1\n\nContent", vector=[], metadata={"header_hierarchy": ["H1"]})]

    # 3. Patch
    with (
        patch("coreason_refinery.pipeline.UnstructuredPdfParser") as MockPdfParser,
        patch("coreason_refinery.pipeline.SemanticChunker") as MockChunker,
    ):
        # Configure Mocks
        parser = MockPdfParser.return_value
        parser.parse.return_value = elements

        chunker = MockChunker.return_value
        chunker.chunk.return_value = chunks

        # 4. Execute
        result = pipeline.process(job)

        # 5. Verify Assertions

        # Parser called with correct path
        parser.parse.assert_called_once_with("/tmp/complex.pdf")

        # Chunker initialized with CORRECT CONFIG from job
        MockChunker.assert_called_once_with(job.config)

        # Chunker called with elements from parser
        chunker.chunk.assert_called_once_with(elements)

        # Result matches chunker output
        assert len(result) == 1
        assert result[0].id == "c1"
        assert result[0].metadata["header_hierarchy"] == ["H1"]
