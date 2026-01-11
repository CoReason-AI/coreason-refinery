# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

import math
from abc import ABC, abstractmethod
from typing import Any, List, Literal

import pandas as pd
from pydantic import BaseModel, Field
from unstructured.documents.elements import (
    Element,
    Footer,
    Header,
    ListItem,
    Table,
    Title,
)
from unstructured.partition.pdf import partition_pdf


class ParsedElement(BaseModel):
    """Represents a single atomic element parsed from a source document.

    This is an intermediate representation before chunking and enrichment.
    """

    text: str
    type: Literal["TITLE", "NARRATIVE_TEXT", "TABLE", "LIST_ITEM", "HEADER", "FOOTER", "UNCATEGORIZED"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedElement]:
        """Parse a file into a list of semantic elements.

        Args:
            file_path: The path to the file to parse.

        Returns:
            A list of ParsedElement objects preserving document order.
        """
        pass  # pragma: no cover


class MockParser(DocumentParser):
    """A mock parser for testing and development purposes."""

    def parse(self, file_path: str) -> List[ParsedElement]:
        """Returns a fixed set of elements for testing."""
        return [
            ParsedElement(
                text="# Clinical Protocol",
                type="TITLE",
                metadata={"page_number": 1},
            ),
            ParsedElement(
                text="1. Introduction",
                type="HEADER",
                metadata={"page_number": 1, "section_depth": 1},
            ),
            ParsedElement(
                text="This is a study about Cisplatin.",
                type="NARRATIVE_TEXT",
                metadata={"page_number": 1},
            ),
            ParsedElement(
                text="| Dose | Response |\n| --- | --- |\n| 10mg | Good |",
                type="TABLE",
                metadata={"page_number": 2, "is_table": True},
            ),
        ]


class UnstructuredPdfParser(DocumentParser):
    """PDF parser using the unstructured library."""

    def parse(self, file_path: str) -> List[ParsedElement]:
        """Parse a PDF file using unstructured."""
        elements = partition_pdf(filename=file_path)
        return [self._map_element(e) for e in elements]

    def _map_element(self, element: Element) -> ParsedElement:
        """Map unstructured element to ParsedElement."""
        element_type = "UNCATEGORIZED"
        if isinstance(element, Title):
            element_type = "TITLE"
        elif isinstance(element, Table):
            element_type = "TABLE"
        elif isinstance(element, ListItem):
            element_type = "LIST_ITEM"
        elif isinstance(element, Header):
            element_type = "HEADER"
        elif isinstance(element, Footer):
            element_type = "FOOTER"
        elif type(element).__name__ == "NarrativeText" or type(element).__name__ == "Text":
            element_type = "NARRATIVE_TEXT"

        # Extract metadata
        metadata = element.metadata.to_dict() if element.metadata else {}

        # Flatten helpful metadata fields to top level of our metadata dict
        if "page_number" in metadata:
            metadata["page_number"] = metadata["page_number"]

        return ParsedElement(
            text=str(element),
            type=element_type,  # type: ignore[arg-type]
            metadata=metadata,
        )


class ExcelParser(DocumentParser):
    """Parses Excel files into Markdown tables using pandas."""

    ROW_LIMIT = 50

    def parse(self, file_path: str) -> List[ParsedElement]:
        """Parse an Excel file.

        Strategy:
            - Load file.
            - Iterate sheets.
            - If sheet rows > ROW_LIMIT, split into chunks.
            - Convert chunks to Markdown tables.
            - Sheet names become Headers.
        """
        # Read all sheets
        try:
            # None reads all sheets as a dict of {sheet_name: df}
            sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            # Handle empty or invalid files gracefully, or re-raise
            # For now, let's assume valid file or allow bubbling up
            raise RuntimeError(f"Failed to parse Excel file: {e}") from e

        elements: List[ParsedElement] = []

        for sheet_name, df in sheets.items():
            # Add Sheet Name as a Header
            elements.append(
                ParsedElement(
                    text=f"Sheet: {sheet_name}",
                    type="HEADER",
                    metadata={"sheet_name": sheet_name, "section_depth": 2},
                )
            )

            if df.empty:
                elements.append(
                    ParsedElement(
                        text="(Empty Sheet)",
                        type="NARRATIVE_TEXT",
                        metadata={"sheet_name": sheet_name},
                    )
                )
                continue

            # Calculate chunks
            total_rows = len(df)
            num_chunks = math.ceil(total_rows / self.ROW_LIMIT)

            for i in range(num_chunks):
                start_idx = i * self.ROW_LIMIT
                end_idx = start_idx + self.ROW_LIMIT

                # Slice the dataframe
                chunk_df = df.iloc[start_idx:end_idx]

                # Convert to markdown
                # index=False usually cleaner for data tables unless index is meaningful
                md_table = chunk_df.to_markdown(index=False, tablefmt="github")

                if md_table:
                    elements.append(
                        ParsedElement(
                            text=md_table,
                            type="TABLE",
                            metadata={
                                "sheet_name": sheet_name,
                                "chunk_index": i,
                                "total_chunks": num_chunks,
                                "row_start": start_idx,
                                "row_end": min(end_idx, total_rows),
                            },
                        )
                    )

        return elements
