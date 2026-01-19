from typing import Any, Dict, List, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """
    Configuration for the ingestion process.

    Attributes:
        use_ocr (bool): Whether to force OCR for PDF processing.
        split_strategy (str): The strategy used for splitting documents (e.g., 'MARKDOWN_HEADER').
    """

    use_ocr: bool = False
    split_strategy: str = "MARKDOWN_HEADER"


class RefinedChunk(BaseModel):
    """
    A semantic chunk of a processed document, enriched with metadata and embeddings.

    Attributes:
        id (str): Unique identifier for the chunk.
        text_markdown (str): The cleaned, markdown-formatted content.
        vector (List[float]): The semantic embedding vector.
        metadata (dict): Contextual metadata (hierarchy, source, table presence, etc.).
    """

    id: str
    text_markdown: str
    vector: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionJob(BaseModel):
    """
    Represents a job to ingest a specific document source.

    Attributes:
        id (UUID): Unique job identifier.
        source_file_path (str): Path or URN to the source file.
        file_type (str): Type of file ('pdf', 'xlsx', 'pptx', 'auto').
        config (IngestionConfig): Configuration parameters for the job.
        status (Literal): Current status of the job.
    """

    id: UUID
    source_file_path: str
    file_type: str = "auto"
    config: IngestionConfig
    status: Literal["PROCESSING", "COMPLETED", "FAILED"]
