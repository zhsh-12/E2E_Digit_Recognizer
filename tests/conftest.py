"""Shared fixtures for all tests."""

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Patch ONNX session before importing api.main (to avoid loading real model)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def _patch_onnx_session():
    """Mock ONNX Runtime session so tests don't need a real quantized model."""
    mock_output = np.array([[0.1, 0.2, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1]],
                           dtype=np.float32)

    mock_session = MagicMock()
    mock_session.run.return_value = [mock_output]

    with patch("api.main.session", mock_session):
        yield


# ---------------------------------------------------------------------------
# Temporary database
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db_path():
    """Create a temporary database file, yield its path, then clean up."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_digit_recognizer.db")
    original = str(Path(__file__).resolve().parent.parent / "digit_recognizer.db")
    yield db_path
    # Clean up
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)


@pytest.fixture
def patch_db_path(temp_db_path):
    """Override common.database.db_path with the temporary path."""
    import common.database as db_mod
    original = db_mod.db_path
    db_mod.db_path = Path(temp_db_path)
    yield
    db_mod.db_path = original


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """Provide a TestClient for the FastAPI app."""
    from api.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image_l():
    """Return a 28×28 grayscale (L mode) PNG image as bytes."""
    img = Image.new("L", (28, 28), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_image_rgb():
    """Return a 28×28 RGB image as bytes."""
    img = Image.new("RGB", (28, 28), color=(64, 128, 192))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_image_rgba():
    """Return a 28×28 RGBA image as bytes."""
    img = Image.new("RGBA", (28, 28), color=(64, 128, 192, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_image_cmyk():
    """Return a 28×28 CMYK image as bytes."""
    img = Image.new("CMYK", (28, 28), color=(0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_image_large():
    """Return a 100×200 grayscale image as bytes (to test resize)."""
    img = Image.new("L", (100, 200), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Sample logits for cal_confidence
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_logits():
    """Return a (1,10) logits array where class 2 has the highest value."""
    return np.array([[0.1, 0.2, 2.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1]],
                    dtype=np.float32)


@pytest.fixture
def uniform_logits():
    """Return a (1,10) logits array where all values are equal."""
    return np.ones((1, 10), dtype=np.float32)
