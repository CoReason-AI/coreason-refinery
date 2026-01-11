# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from abc import ABC, abstractmethod
from typing import Any, List, Literal

from pydantic import BaseModel, Field


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
