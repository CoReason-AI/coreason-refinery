# The Architecture and Utility of coreason-refinery

### 1. The Philosophy (The Why)

In the era of Retrieval-Augmented Generation (RAG), the adage "garbage in, garbage out" has evolved into "structure in, reason out." Standard ingestion pipelines treat documents as flat streams of characters, blindly chopping them into 500-token chunks. This approach is catastrophic for scientific and regulatory data. A dosage table split in half loses its meaning; a safety warning separated from its protocol header becomes a dangerous hallucination.

**coreason-refinery** was built on the conviction that **Structure is Prerequisite to Reason**. It acts as an industrial processing plant for data, sitting between raw chaos (PDFs, Spreadsheets) and the Semantic Archive. Its primary mission is not just to read text, but to preserve the *logic* of the document. By enforcing "Context Injection"—where every snippet of text carries its ancestral hierarchy (e.g., "Protocol > Toxicity > Grade 4")—it ensures that downstream LLMs understand not just *what* was said, but *where* and *why*.

### 2. Under the Hood (The Dependencies & logic)

The package orchestrates a specialized, multi-modal stack to treat different file types according to their nature.

*   **`marker-pdf` (The Layout Engine):** For high-fidelity PDF processing, the architecture prioritizes Deep Learning Computer Vision. Unlike traditional scrapers, it "sees" the document layout, preserving complex grids and mathematical notation.
*   **`unstructured` (The Universal Fallback):** Used for legacy formats and as a robust fallback, ensuring that no document is left behind, even if it requires simpler text extraction.
*   **`pandas` & `openpyxl` (The Relational Layer):** Instead of treating spreadsheets as text, the `ExcelParser` leverages `pandas` to respect their grid nature. It intelligently chunks large sheets by row groups, repeating column headers for every segment so context is never lost.
*   **`pydantic` (The Contract):** Strict data models (`IngestionJob`, `RefinedChunk`) ensure that data flowing through the refinery is typed, validated, and predictable.

**The Semantic Segmenter:**
The heart of the system is the `SemanticChunker`. Unlike naive splitters, it uses a "rolling context" mechanism. It maintains a stack of active headers (the "breadcrumbs") as it traverses the document. When it cuts a chunk, it doesn't just save the text; it prepends the entire active hierarchy. Furthermore, it possesses "Table Awareness"—it will deliberately violate length limits to keep a table intact, preventing the common failure mode where row data is separated from its column headers.

### 3. In Practice (The How)

The following examples demonstrate how `coreason-refinery` transforms raw files into context-aware semantic chunks.

#### The Happy Path: Processing a Clinical Protocol

In this scenario, we ingest a PDF. The pipeline automatically detects the file type, selects the appropriate parser (prioritizing layout analysis), and attaches the section hierarchy to every output chunk.

```python
import uuid
from coreason_refinery.models import IngestionJob, IngestionConfig
from coreason_refinery.pipeline import RefineryPipeline

# 1. Configure the Job
# We define a job for a PDF protocol, requesting semantic splitting.
job = IngestionJob(
    id=uuid.uuid4(),
    source_file_path="data/clinical_protocol_v2.pdf",
    file_type="pdf",
    config=IngestionConfig(
        chunk_strategy="HEADER",
        segment_len=500
    ),
    status="PROCESSING"
)

# 2. Run the Refinery
pipeline = RefineryPipeline()
chunks = pipeline.process(job)

# 3. Inspect the Semantic Output
# The pipeline has not just extracted text, but enriched it.
first_chunk = chunks[0]

print(f"Chunk ID: {first_chunk.id}")
print(f"Context: {first_chunk.metadata.get('header_hierarchy')}")
# Output: ['Protocol 101', 'Section 4: Safety', '4.1 Adverse Events']

print(f"Content:\n{first_chunk.text[:100]}...")
# Output: Context: Protocol 101 > Section 4: Safety > 4.1 Adverse Events
#
#         The frequency of adverse events was monitored...
```

#### Handling Structured Data: The Excel Rescue

When processing spreadsheets, `RefineryPipeline` uses the `ExcelParser` to ensure large datasets are broken down without losing their schema.

```python
# 1. Ingest a large Lab Data spreadsheet
excel_job = IngestionJob(
    id=uuid.uuid4(),
    source_file_path="data/lab_results.xlsx",
    file_type="excel",  # or "auto"
    config=IngestionConfig(segment_len=1000),
    status="PROCESSING"
)

# 2. Process
excel_chunks = pipeline.process(excel_job)

# 3. Verify Table Integrity
# Each chunk represents a slice of rows, formatted as a clean Markdown table.
for chunk in excel_chunks:
    if "chunk_index" in chunk.metadata:
        print(f"Sheet: {chunk.metadata['sheet_name']} | Rows: {chunk.metadata['row_start']}-{chunk.metadata['row_end']}")
        # The text is a Markdown table, ready for LLM consumption
        # | PatientID | Test | Result |
        # |-----------|------|--------|
        # | 001       | WBC  | 5.4    |
```
