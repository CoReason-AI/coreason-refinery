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
from typing import Dict, List, Type

from coreason_refinery.models import ParsedElement


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    Each format-specific parser must implement the `parse` method.
    """

    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedElement]:
        """
        Parse a document into a list of semantic elements.

        Args:
            file_path (str): The path to the file to parse.

        Returns:
            List[ParsedElement]: A list of parsed elements (titles, text, tables, etc.).
        """
        pass  # pragma: no cover


class ParserRegistry:
    """
    Registry for managing and retrieving document parsers based on file extensions.
    """

    _registry: Dict[str, Type[DocumentParser]] = {}

    @classmethod
    def register(cls, extension: str, parser_cls: Type[DocumentParser]) -> None:
        """
        Register a parser for a specific file extension.

        Args:
            extension (str): The file extension (e.g., ".pdf"). Case-insensitive.
            parser_cls (Type[DocumentParser]): The parser class to handle the extension.
        """
        cls._registry[extension.lower()] = parser_cls

    @classmethod
    def get_parser(cls, extension: str) -> Type[DocumentParser]:
        """
        Retrieve a parser class for the given extension.

        Args:
            extension (str): The file extension (e.g., ".pdf").

        Returns:
            Type[DocumentParser]: The registered parser class.

        Raises:
            ValueError: If no parser is registered for the extension.
        """
        extension = extension.lower()
        if extension not in cls._registry:
            raise ValueError(f"No parser registered for extension: {extension}")
        return cls._registry[extension]
