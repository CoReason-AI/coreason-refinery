# coreason-refinery

![License: Prosperity 3.0](https://img.shields.io/badge/License-Prosperity%203.0-blue?style=flat&label=License&color=blue)
![Build Status](https://github.com/CoReason-AI/coreason_refinery/actions/workflows/ci.yml/badge.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

**coreason-refinery** is the industrial processing plant for data ingestion at CoReason-AI. It transforms raw documents (PDF, Excel, PPTX) into semantically structured, machine-readable Markdown, ready for RAG (Retrieval-Augmented Generation) pipelines.

Unlike traditional RAG tools that blindly chunk text, **coreason-refinery** is **Structure-Aware**. It preserves the logical hierarchy of documents (headers, sections) and ensures complex artifacts like tables and equations are kept intact.

## Features

*   **Multi-Modal Parsing**:
    *   **PDF**: Vision-first parsing to preserve table grids and layout.
    *   **Excel/CSV**: Treated as relational data, converted to Markdown tables with header preservation.
    *   **PowerPoint**: Flattens slides into linear narratives (planned support for speaker notes).
*   **Semantic Segmentation**:
    *   Splits documents by logical headers (#, ##) rather than character counts.
    *   **Table Rescue**: Never splits tables mid-row; merges tables spanning multiple pages.
*   **Context Injection ("Breadcrumbs")**:
    *   Enriches every chunk with its hierarchical path (e.g., `Context: Protocol > Section 4 > Toxicity`).
    *   Ensures LLMs understand the specific scope of any given text snippet.
*   **GxP Compliance Ready**:
    *   Designed for lineage tracking and metadata enrichment.

## Installation

```bash
pip install coreason-refinery
```

*Note: This package requires Python 3.12+.*

## Usage

Here is how to initialize and run a refinery job:

```python
import uuid
from coreason_refinery.models import IngestionJob, IngestionConfig
from coreason_refinery.pipeline import RefineryPipeline

# 1. Configure the Job
config = IngestionConfig(
    chunk_strategy="HEADER",
    segment_len=500
)

job = IngestionJob(
    id=uuid.uuid4(),
    source_file_path="path/to/document.pdf",
    file_type="auto",  # Infers PDF/Excel/CSV
    config=config,
    status="PROCESSING"
)

# 2. Run the Pipeline
pipeline = RefineryPipeline()
chunks = pipeline.process(job)

# 3. Inspect Results
for chunk in chunks:
    print(f"--- Chunk ID: {chunk.id} ---")
    print(chunk.text)
    print(f"Metadata: {chunk.metadata}")
```

## Architecture

The pipeline consists of three main stages:

1.  **The Cracker (Parsing)**: Routes files to specialized parsers (`UnstructuredPdfParser`, `ExcelParser`) to extract atomic elements.
2.  **The Cutter (Segmentation)**: Reassembles elements into `RefinedChunk`s based on document structure, applying "Rolling Context" to preserve hierarchy.
3.  **The Enricher (Metadata)**: (Planned) Adds lineage and semantic tags.

## License

This software is proprietary and dual-licensed under the **Prosperity Public License 3.0**.
Commercial use beyond a 30-day trial requires a separate license.
See `LICENSE` for details.
