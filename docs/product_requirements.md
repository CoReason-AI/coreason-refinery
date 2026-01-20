# **Product Requirements Document: coreason-refinery**

Domain: Data Engineering, ETL, & RAG Preparation
Architectural Role: The "Refinery" / Ingestion Engine
Core Philosophy: "Structure is Prerequisite to Reason. A Table is a Database. Context is King."
Dependencies: marker-pdf (Layout Engine), unstructured (Legacy Formats), coreason-validator (Schema), coreason-mcp (Source), coreason-aegis (PII Sanitization)

## **\---**

**1\. Executive Summary**

coreason-refinery is the industrial processing plant for data ingestion. It sits between the raw data sources and the Semantic Archive.

Its mandate is **Semantic Structuring**. Traditional RAG pipelines fail on scientific documents because they blindly chunk text. coreason-refinery is **Structure-Aware**. It uses **Deep Learning Computer Vision (Marker)** to "see" PDF layouts, and specialized parsers for Excel and PowerPoint. It ensures that complex artifacts (Dosage Tables, Slide Decks, Equations) are converted into clean, machine-readable Markdown.

Crucially, it implements **Context Injection**: every chunk is enriched with its hierarchy (e.g., "Protocol \> Section 4 \> Toxicity"), ensuring the LLM understands the specific scope of the text.

## **2\. Functional Philosophy**

The agent must implement the **Vision-Structure-Enrich-Vectorize Loop**:

1. **Vision-First Parsing (SOTA):** We prioritize layout analysis. A PDF is processed by **Marker** to preserve table grids and mathematical notation.
2. **Format Agnosticism:** While PDFs use Vision, Excel files are treated as Relational Data, and PowerPoints are flattened into linear narratives. All paths lead to a unified **Markdown** representation.
3. **Semantic Segmentation:** We segment by **Document Logic** (Headers, Table Boundaries), not arbitrary character counts.
4. **Metadata & Lineage:** Every segment carries GxP lineage (Source, Version, Author).
5. **PII Scrubbing (Air Gap):** Data is sanitized by coreason-aegis *before* embedding.

## **\---**

**3\. Core Functional Requirements (Component Level)**

### **3.1 The Multi-Modal Parser (The Cracker)**

**Concept:** A routing engine that selects the best extractor for the file type.

* **PDF Pipeline (Primary):**
  * **Engine:** **marker-pdf**.
  * **Capability:** Extracts high-fidelity tables, equations (LaTeX), and headers.
  * **Fallback:** If Marker fails (or for simple text PDFs), fall back to unstructured (fast mode).
* **Spreadsheet Pipeline (Excel/CSV):**
  * **Engine:** pandas / openpyxl.
  * **Logic:** Converts sheets into **Markdown Tables**. Preserves column headers.
  * **Constraint:** Large sheets (\>50 rows) are summarized or split by row groups, ensuring headers repeat.
* **Presentation Pipeline (PowerPoint):**
  * **Engine:** unstructured.
  * **Logic:** Flattens slides into a linear narrative. **Requirement:** Must extract *Speaker Notes* and append them as context for the slide content.

### **3.2 The Semantic Segmenter (The Cutter)**

**Concept:** Intelligent segmentation based on the Markdown hierarchy.

* **Header-Based Splitting:** Detects \#\# headers to create logical sections.
* **Table Preservation:** A Table must *never* be split mid-row. If it spans pages, it is merged into one logical block.
* **Rolling Context (The "Breadcrumbs"):**
  * **Logic:** If a segment is extracted from "Section 4.1.2", the text of Headers 4, 4.1, and 4.1.2 is prepended to the chunk.
  * **Value:** Ensures a chunk reading "Stop treatment" is embedded as "Safety \> Toxicity \> Grade 4 \> Stop treatment".

### **3.3 The Metadata Enricher (The Labeler)**

**Concept:** Adds GxP context to vectors.

* **Extraction:** Pulls file properties (Author, Created Date, Version ID).
* **Inference:** Uses a lightweight LLM to infer semantic tags (Therapeutic Area, Drug Name).
* **Validation:** Calls coreason-validator to ensure tags match the controlled vocabulary.

### **3.4 The PII Scrubber (The Filter)**

**Concept:** Sanitization layer.

* **Integration:** Calls coreason-aegis to detect/redact PII.
* **Audit:** Logs the redaction event to coreason-veritas.

## **\---**

**4\. Integration Requirements (The Ecosystem)**

* **Source (coreason-mcp):** Consumes data streams.
* **Embedding (coreason-search):** Uses Qwen-32k embeddings via coreason-search.
* **Destination (coreason-archive):** Writes the (Vector, Metadata, Markdown) tuple.
* **Governance (coreason-veritas):** Logs the "Ingestion Manifest".

## **\---**

**5\. User Stories (Behavioral Expectations)**

### **Story A: The "Table Rescue" (Structure Preservation)**

Context: User uploads a PDF with a multi-page dosing table.
Action: Marker detects the grid, merges headers across pages, and outputs one clean Markdown table. Segmenter keeps it intact.
Result: Agent accurately answers complex lookup questions about dosing.

### **Story B: The "Header Context" (Semantic Context)**

Context: A chunk simply says: "Stop treatment immediately."
Action: Refinery prepends the hierarchy: Context: \[Protocol 999\] \> \[Section 4: Toxicity\] \> \[Grade 4 Events\].
Result: Vector search retrieves this only when the user asks about toxicity in Protocol 999, avoiding dangerous confusion.

### **Story C: The "Math Recognition" (Scientific Fidelity)**

Context: A Pharmacometrician uploads a paper with PK/PD equations.
Action: Marker outputs LaTeX: $$CL \= \\frac{Dose}{AUC}$$.
Result: The LLM can interpret and render the math correctly.

### **Story D: The "Ghost Data" (Atomic Versioning)**

Context: "Protocol v1.1" replaces "Protocol v1.0".
Action: Refinery detects the ID match, purges v1.0 vectors, and ingests v1.1.
Result: No conflicting advice from obsolete documents.

## **\---**

**6\. Data Schema**

### **IngestionJob**

Python

class IngestionJob(BaseModel):
    id: UUID
    source\_urn: str            \# "mcp://sharepoint/protocol\_v2.pdf"
    file\_type: str             \# "pdf", "xlsx", "pptx"
    config: IngestionConfig    \# { use\_ocr: True, split\_strategy: "MARKDOWN\_HEADER" }
    status: Literal\["PROCESSING", "COMPLETED", "FAILED"\]

### **RefinedSegment**

Python

class RefinedSegment(BaseModel):
    id: str                     \# Unique Vector ID
    text\_markdown: str          \# The Cleaned Content
    vector: List\[float\]         \# Embedding

    metadata: dict \= {
        "source\_urn": "mcp://sharepoint/doc.pdf",
        "hierarchy\_path": \["Safety", "Toxicity"\], \# The Rolling Context
        "contains\_table": True,
        "contains\_equation": False,
        "speaker\_notes\_included": False \# For PPTX
    }

## **\---**

**7\. Implementation Directives for the Coding Agent**

1. **Library Selection:**
   * **PDF:** marker-pdf (Primary), unstructured (Fallback).
   * **Office:** openpyxl (Excel), python-pptx (PowerPoint).
2. **Resource Management (Hybrid Compute):**
   * **GPU:** Route PDF/Marker jobs to GPU instances.
   * **CPU/Parallel:** Route Excel/PPTX jobs and text cleaning tasks to multi-core CPU instances. Use multiprocessing to handle page-level parallelism where possible.
3. **Markdown as Lingua Franca:** Ensure all parsers output strictly formatted Markdown.
4. **Error Resilience:** Implement page-level error handling. A single bad page should not fail the entire job; log it and continue.
