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
def mock_read_excel() -> Generator[MagicMock, None, None]:
    with patch("pandas.read_excel") as mock:
        yield mock


def test_excel_parser_simple_sheet(mock_read_excel: MagicMock) -> None:
    """Test parsing a simple single-sheet Excel file."""
    # Setup mock DF
    data = {"Col1": ["A", "B"], "Col2": [1, 2]}
    df = pd.DataFrame(data)

    # Mock return: dict of sheet_name -> df
    mock_read_excel.return_value = {"Sheet1": df}

    parser = ExcelParser()
    elements = parser.parse("dummy.xlsx")

    assert len(elements) == 2
    # 1. Header (Sheet name)
    assert elements[0].type == "HEADER"
    assert elements[0].text == "Sheet: Sheet1"

    # 2. Table
    assert elements[1].type == "TABLE"
    # Note: tabulate may add extra spaces/formatting
    assert "Col1" in elements[1].text
    assert "Col2" in elements[1].text
    assert "A" in elements[1].text
    assert "1" in elements[1].text


def test_excel_parser_large_sheet(mock_read_excel: MagicMock) -> None:
    """Test splitting a large sheet (>50 rows)."""
    # Create 110 rows
    data = {"ID": range(110)}
    df = pd.DataFrame(data)

    mock_read_excel.return_value = {"LargeSheet": df}

    parser = ExcelParser()
    elements = parser.parse("dummy.xlsx")

    # Expect:
    # 1. Header (Sheet name)
    # 2. Table Chunk 1 (rows 0-50)
    # 3. Table Chunk 2 (rows 50-100)
    # 4. Table Chunk 3 (rows 100-110)
    assert len(elements) == 4

    assert elements[0].text == "Sheet: LargeSheet"

    # Check chunks
    chunk1 = elements[1]
    assert chunk1.type == "TABLE"
    assert chunk1.metadata["chunk_index"] == 0
    assert chunk1.metadata["row_start"] == 0
    # verify content roughly
    assert "0" in chunk1.text

    chunk3 = elements[3]
    assert chunk3.type == "TABLE"
    assert chunk3.metadata["chunk_index"] == 2
    assert chunk3.metadata["row_start"] == 100
    assert "109" in chunk3.text


def test_excel_parser_empty_sheet(mock_read_excel: MagicMock) -> None:
    """Test handling of empty sheets."""
    df = pd.DataFrame()
    mock_read_excel.return_value = {"EmptySheet": df}

    parser = ExcelParser()
    elements = parser.parse("dummy.xlsx")

    assert len(elements) == 2
    assert elements[0].text == "Sheet: EmptySheet"
    assert elements[1].type == "NARRATIVE_TEXT"
    assert elements[1].text == "(Empty Sheet)"


def test_excel_parser_multiple_sheets(mock_read_excel: MagicMock) -> None:
    """Test parsing multiple sheets."""
    df1 = pd.DataFrame({"A": [1]})
    df2 = pd.DataFrame({"B": [2]})

    mock_read_excel.return_value = {"Sheet1": df1, "Sheet2": df2}

    parser = ExcelParser()
    elements = parser.parse("dummy.xlsx")

    # Sheet1 Header, Sheet1 Table, Sheet2 Header, Sheet2 Table
    assert len(elements) == 4
    assert elements[0].text == "Sheet: Sheet1"
    assert elements[2].text == "Sheet: Sheet2"


def test_excel_parser_error(mock_read_excel: MagicMock) -> None:
    """Test error handling."""
    mock_read_excel.side_effect = ValueError("Invalid file")

    parser = ExcelParser()
    with pytest.raises(RuntimeError) as excinfo:
        parser.parse("bad.xlsx")

    assert "Failed to parse Excel file" in str(excinfo.value)
