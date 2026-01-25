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
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, Request, UploadFile

from coreason_refinery.models import IngestionConfig, IngestionJob, RefinedChunk
from coreason_refinery.pipeline import RefineryPipelineAsync
from coreason_refinery.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initializing the pipeline."""
    logger.info("Initializing Refinery Pipeline...")
    pipeline = RefineryPipelineAsync()
    async with pipeline:
        app.state.pipeline = pipeline
        yield
    logger.info("Refinery Pipeline shutdown.")


app = FastAPI(lifespan=lifespan, title="Coreason Refinery", version="0.1.0")


@app.post("/ingest", response_model=List[RefinedChunk])
async def ingest(request: Request, file: UploadFile) -> List[RefinedChunk]:
    """Ingest a file and return refined chunks."""
    logger.info(f"Received ingestion request for file: {file.filename}")

    # Create a temporary file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    tmp_path = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            # Read and write in chunks to avoid loading large files into memory
            # and to remain async-friendly
            CHUNK_SIZE = 1024 * 1024  # 1MB
            while chunk := await file.read(CHUNK_SIZE):
                tmp_file.write(chunk)

        job_id = uuid.uuid4()
        job = IngestionJob(
            id=job_id, source_file_path=tmp_path, file_type="auto", config=IngestionConfig(), status="PROCESSING"
        )

        pipeline: RefineryPipelineAsync = request.app.state.pipeline
        chunks = await pipeline.process(job)
        return chunks

    except Exception as e:
        logger.exception("Ingestion failed")
        raise e
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"Removed temporary file: {tmp_path}")
            except Exception:
                logger.warning(f"Failed to remove temporary file: {tmp_path}")
