import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import List

from coreason_validator.schemas.knowledge import ArtifactType, KnowledgeArtifact
from fastapi import FastAPI, File, Request, UploadFile

from coreason_refinery.models import IngestionConfig, IngestionJob, RefinedChunk
from coreason_refinery.pipeline import RefineryPipelineAsync


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Initialize the pipeline
    async with RefineryPipelineAsync() as pipeline:
        app.state.pipeline = pipeline
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/ingest", response_model=List[KnowledgeArtifact])
async def ingest_file(request: Request, file: UploadFile = File(...)) -> List[KnowledgeArtifact]:  # noqa: B008
    # Save the uploaded file to a temporary directory
    # We use delete=False so we can pass the path to the pipeline, then delete it manually
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Create IngestionJob
        job_id = uuid.uuid4()
        job = IngestionJob(
            id=job_id, source_file_path=tmp_path, file_type="auto", config=IngestionConfig(), status="PROCESSING"
        )

        # Process the job asynchronously
        # Using the pipeline stored in app.state
        chunks: List[RefinedChunk] = await request.app.state.pipeline.process(job)

        # Transform RefinedChunks into KnowledgeArtifacts
        artifacts = []
        for chunk in chunks:
            artifact = KnowledgeArtifact(
                content=chunk.text,
                source_location=chunk.metadata,
                source_urn=f"urn:job:{job_id}:file:{file.filename}",
                artifact_type=ArtifactType.TEXT,
            )
            artifacts.append(artifact)

        return artifacts

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
