"""Unit tests for api/services.py — direct function tests for missing coverage."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile


@pytest.mark.asyncio
async def test_read_upload_file_exception():
    """_read_upload_file: when file.read() raises, error dict is returned."""
    from api.services import _read_upload_file

    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "bad.png"
    mock_file.read.side_effect = Exception("Read failed")

    result = await _read_upload_file(mock_file)
    assert result["filename"] == "bad.png"
    assert result["contents"] is None
    assert result["error"] == "Read failed"


@pytest.mark.asyncio
async def test_run_inference_concurrently_with_read_error():
    """run_inference_concurrently: file read error triggers error branch."""
    from api.services import _read_upload_file, run_inference_concurrently

    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "bad.png"
    mock_file.read.side_effect = Exception("Read failed")

    with patch("api.services._read_upload_file", wraps=_read_upload_file):
        results = await run_inference_concurrently([mock_file])
        assert len(results) == 1
        assert results[0]["filename"] == "bad.png"
        assert results[0]["error"] == "Read failed"
