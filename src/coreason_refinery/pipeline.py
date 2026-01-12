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
from typing import List

from coreason_refinery.chunking import SemanticChunker
from coreason_refinery.models import IngestionJob, RefinedChunk
from coreason_refinery.parsing import (
    DocumentParser,
    ExcelParser,
    UnstructuredPdfParser,
)
from coreason_refinery.utils.logger import logger


class RefineryPipeline:
    """Orchestrates the ingestion process: Parsing -> Chunking -> Enrichment."""

    def process(self, job: IngestionJob) -> List[RefinedChunk]:
        """Process an ingestion job.

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
            elements = parser.parse(job.source_file_path)
            logger.info(f"Parsed {len(elements)} elements")

            # 3. Chunk Elements
            chunker = SemanticChunker(job.config)
            logger.debug("Chunking elements")
            chunks = chunker.chunk(elements)
            logger.info(f"Generated {len(chunks)} chunks")

            # 4. Enrichment (Placeholder for future atomic unit)
            # chunks = self._enrich(chunks)

            # Update job status (conceptually - though job object is passed by value here usually)
            # In a real system, we'd update a DB. Here we just return the chunks.

            return chunks

        except Exception as e:
            logger.exception(f"Processing failed for job {job.id}")
            raise RuntimeError(f"Pipeline processing failed: {e}") from e

    def _get_parser(self, job: IngestionJob) -> DocumentParser:
        """Select the appropriate parser based on file type."""
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
