"""Shared pytest fixtures for all test modules."""
import pytest
import torch
import torch.nn as nn
import io
from PIL import Image
from unittest.mock import patch, MagicMock
from pathlib import Path


# ================================================================
# Helper: create a fake test image as PNG bytes
# ================================================================

def make_test_image_bytes(mode: str = "L", size=(28, 28), color=128) -> bytes:
    """Create a simple test image and return its PNG bytes."""
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ================================================================
# Mock model: a simple nn.Module that returns fixed logits
# Used to avoid loading real .pth weights in tests
# ================================================================

class MockDigitModel(nn.Module):
    """A fake model that returns predetermined logits for test assertions."""

    def __init__(self, fixed_prediction: int = 2):
        super().__init__()
        self.fixed_prediction = fixed_prediction

    def forward(self, x):
        """Return a batch of logits with fixed_prediction as the max."""
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 10)
        logits[:, self.fixed_prediction] = 10.0
        return logits

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def mock_model():
    """Fixture providing a MockDigitModel that always predicts 2."""
    model = MockDigitModel(fixed_prediction=2)
    with patch("app.model.get_model", return_value=model):
        with patch("app.preprocess.set_device", return_value=torch.device("cpu")):
            yield model


# ================================================================
# Mock database: use a temp file via monkeypatch
# ================================================================

@pytest.fixture(scope="function")
def mock_db(tmp_path, monkeypatch):
    """Point common.config.db_path to a fresh temp file per test function."""
    test_db = tmp_path / "test.db"
    monkeypatch.setattr("common.config.db_path", test_db)
    yield test_db
    if test_db.exists():
        test_db.unlink()
