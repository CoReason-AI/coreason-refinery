# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

import os
from typing import Any, List, Optional

import anyio
import httpx

from coreason_refinery.models import IngestionJob, RefinedChunk
from coreason_refinery.parsing import (
    DocumentParser,
    ExcelParser,
    UnstructuredPdfParser,
)
from coreason_refinery.segmentation import SemanticChunker
from coreason_refinery.utils.logger import logger


class RefineryPipelineAsync:
    """Async Core of the Ingestion Pipeline.

    Handles all logic with strict lifecycle management.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        """Initialize the pipeline with an optional HTTP client.

        Args:
            client: Optional httpx.AsyncClient for connection pooling.
        """
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

    async def __aenter__(self) -> "RefineryPipelineAsync":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self._internal_client:
            await self._client.aclose()
        # Close other resources if any

    async def process(self, job: IngestionJob) -> List[RefinedChunk]:
        """Process an ingestion job asynchronously.

        Executes the Vision-Structure-Enrich-Vectorize Loop (partially).
        1. Selects Parser (Multi-Modal Parser).
        2. Parses Document into atomic elements (offloaded to thread).
        3. Chunks Elements (Semantic Segmenter) into RefinedChunks (offloaded to thread).

        Args:
            job: The ingestion job configuration.

        Returns:
            A list of RefinedChunk objects ready for downstream consumption.

        Raises:
            ValueError: If the file type is not supported.
            RuntimeError: If processing fails.
        """
        logger.info(f"Starting processing for job {job.id} (File: {job.source_file_path})")

        try:
            # 1. Select Parser
            parser = self._get_parser(job)

            # 2. Parse Document
            logger.debug(f"Parsing file using {type(parser).__name__}")
            # Offload blocking I/O and CPU bound parsing to a worker thread
            elements = await anyio.to_thread.run_sync(parser.parse, job.source_file_path)
            logger.info(f"Parsed {len(elements)} elements")

            # 3. Chunk Elements
            chunker = SemanticChunker(job.config)
            logger.debug("Chunking elements")
            # Offload CPU bound chunking to a worker thread
            chunks = await anyio.to_thread.run_sync(chunker.chunk, elements)
            logger.info(f"Generated {len(chunks)} chunks")

            # 4. Enrichment (Placeholder for future atomic unit)
            # chunks = await self._enrich(chunks)

            # Cast to ensure mypy knows this is a list of RefinedChunk
            return chunks  # type: ignore[no-any-return]

        except Exception as e:
            logger.exception(f"Processing failed for job {job.id}")
            raise RuntimeError(f"Pipeline processing failed: {e}") from e

    def _get_parser(self, job: IngestionJob) -> DocumentParser:
        """Select the appropriate parser based on file type.

        Acts as the routing engine ('The Cracker').

        Args:
            job: The ingestion job containing file info.

        Returns:
            An instance of a DocumentParser subclass.

        Raises:
            ValueError: If file type is unsupported.
        """
        file_type = job.file_type.lower()

        # If file_type is generic or auto, try to deduce from extension
        if file_type == "auto":
            _, ext = os.path.splitext(job.source_file_path)
            ext = ext.lower().lstrip(".")
            if ext in ["pdf"]:
                file_type = "pdf"
            elif ext in ["xlsx", "xls", "csv"]:
                file_type = "xlsx"  # Treat CSV as Excel for now per PRD 3.1
            else:
                # Default fallback or error
                pass

        if file_type == "pdf":
            return UnstructuredPdfParser()
        elif file_type in ["xlsx", "excel"]:
            return ExcelParser()
        # elif file_type == "pptx":
        #     return PptxParser()
        else:
            raise ValueError(f"Unsupported file type: {job.file_type}")


class RefineryPipeline:
    """Sync Facade for RefineryPipelineAsync.

    Orchestrates the ingestion process: Parsing -> Chunking -> Enrichment.
    Wraps the Async Core using anyio.run.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self._async = RefineryPipelineAsync(client=client)

    def __enter__(self) -> "RefineryPipeline":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        anyio.run(self._async.__aexit__, exc_type, exc_val, exc_tb)

    def process(self, job: IngestionJob) -> List[RefinedChunk]:
        """Process an ingestion job synchronously.

        Args:
            job: The ingestion job configuration.

        Returns:
            A list of RefinedChunk objects.
        """
        # anyio.run is generic, cast return value
        return anyio.run(self._async.process, job)  # type: ignore[no-any-return]
