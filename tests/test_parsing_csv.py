# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_refinery

from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from coreason_refinery.parsing import ExcelParser


@pytest.fixture
def mock_read_csv() -> Generator[MagicMock, None, None]:
    with patch("pandas.read_csv") as mock:
        yield mock


def test_csv_parser_simple(mock_read_csv: MagicMock) -> None:
    """Test parsing a simple CSV file."""
    # Setup mock DF
    data = {"Col1": ["A", "B"], "Col2": [1, 2]}
    df = pd.DataFrame(data)

    # Mock return: df directly (read_csv returns df, not dict)
    mock_read_csv.return_value = df

    parser = ExcelParser()
    elements = parser.parse("dummy.csv")

    assert len(elements) == 2
    # 1. Header (Wrapper Sheet name)
    assert elements[0].type == "HEADER"
    assert elements[0].text == "Sheet: CSV_Data"

    # 2. Table
    assert elements[1].type == "TABLE"
    assert "Col1" in elements[1].text
    assert "A" in elements[1].text


def test_csv_parser_empty(mock_read_csv: MagicMock) -> None:
    """Test parsing an empty CSV file."""
    df = pd.DataFrame()
    mock_read_csv.return_value = df

    parser = ExcelParser()
    elements = parser.parse("empty.csv")

    assert len(elements) == 2
    assert elements[0].text == "Sheet: CSV_Data"
    assert elements[1].type == "NARRATIVE_TEXT"
    assert elements[1].text == "(Empty Sheet)"


def test_csv_parser_large(mock_read_csv: MagicMock) -> None:
    """Test splitting a large CSV file."""
    # Create 60 rows
    data = {"ID": range(60)}
    df = pd.DataFrame(data)

    mock_read_csv.return_value = df

    parser = ExcelParser()
    elements = parser.parse("large.csv")

    # Header + 2 Chunks
    assert len(elements) == 3
    assert elements[0].text == "Sheet: CSV_Data"
    assert elements[1].metadata["chunk_index"] == 0
    assert elements[2].metadata["chunk_index"] == 1


def test_csv_parser_error(mock_read_csv: MagicMock) -> None:
    """Test error handling for CSV."""
    mock_read_csv.side_effect = FileNotFoundError("File not found")

    parser = ExcelParser()
    with pytest.raises(RuntimeError) as excinfo:
        parser.parse("missing.csv")

    assert "Failed to parse Structured file (missing.csv)" in str(excinfo.value)


def test_csv_parser_encoding_error(mock_read_csv: MagicMock) -> None:
    """Test handling of UnicodeDecodeError (simulating encoding issues)."""
    mock_read_csv.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")

    parser = ExcelParser()
    with pytest.raises(RuntimeError) as excinfo:
        parser.parse("latin1.csv")

    assert "Failed to parse Structured file (latin1.csv)" in str(excinfo.value)
    assert "invalid start byte" in str(excinfo.value)


def test_csv_parser_malformed(mock_read_csv: MagicMock) -> None:
    """Test handling of pandas ParserError (malformed CSV)."""
    mock_read_csv.side_effect = pd.errors.ParserError("Error tokenizing data")

    parser = ExcelParser()
    with pytest.raises(RuntimeError) as excinfo:
        parser.parse("bad_format.csv")

    assert "Failed to parse Structured file (bad_format.csv)" in str(excinfo.value)


def test_csv_parser_empty_data_error(mock_read_csv: MagicMock) -> None:
    """Test handling of pandas EmptyDataError."""
    # EmptyDataError is raised by pandas when a file is empty or has no columns
    mock_read_csv.side_effect = pd.errors.EmptyDataError("No columns to parse from file")

    parser = ExcelParser()
    with pytest.raises(RuntimeError) as excinfo:
        parser.parse("no_columns.csv")

    assert "Failed to parse Structured file (no_columns.csv)" in str(excinfo.value)


def test_csv_newlines_in_quotes(mock_read_csv: MagicMock) -> None:
    """Test that newlines within quoted fields are preserved in the Markdown table output."""
    # Dataframe simulating:
    # Col1,Col2
    # "Line 1\nLine 2",Value
    data = {"Col1": ["Line 1\nLine 2"], "Col2": ["Value"]}
    df = pd.DataFrame(data)
    mock_read_csv.return_value = df

    parser = ExcelParser()
    elements = parser.parse("complex.csv")

    assert len(elements) == 2
    table_text = elements[1].text

    # Check that "Line 1" and "Line 2" both appear.
    # Note: df.to_markdown() behavior on newlines depends on tablefmt.
    # 'github' format usually keeps them as raw \n or converts to <br> depending on pandas version/tabulate.
    # We verify the content is present.
    assert "Line 1" in table_text
    assert "Line 2" in table_text
