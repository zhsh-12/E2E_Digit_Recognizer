"""Tests for app/preprocess.py — image preprocessing and device detection."""
import pytest
import torch
from PIL import Image
import io

from app.preprocess import set_device, transform


class TestSetDevice:
    """set_device should always return a valid torch.device."""

    def test_returns_torch_device(self):
        device = set_device()
        assert isinstance(device, torch.device)

    def test_device_string_is_valid(self):
        device = set_device()
        assert device.type in ("cpu", "cuda", "mps")

class TestSetDeviceAdditional:
    """Additional set_device tests via mocking torch.backends."""

    def test_mps_branch(self, monkeypatch):
        """When mps is available and cuda is not, set_device returns mps."""
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        device = set_device()
        assert device.type == "mps"

    def test_cuda_branch(self, monkeypatch):
        """When mps is not available but cuda is, set_device returns cuda."""
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        device = set_device()
        assert device.type == "cuda"

    def test_cpu_branch(self, monkeypatch):
        """When neither mps nor cuda is available, set_device returns cpu."""
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        device = set_device()
        assert device.type == "cpu"



class TestTransform:
    """transform should handle L, RGB, RGBA modes correctly."""

    @staticmethod
    def _make_image(mode: str, size=(28, 28), color=128):
        """Helper to create a simple PIL image."""
        if mode == "L":
            return Image.new("L", size, color=color)
        elif mode == "RGB":
            return Image.new("RGB", size, color=(color, color, color))
        elif mode == "RGBA":
            return Image.new("RGBA", size, color=(color, color, color, 255))
        return Image.new(mode, size)

    def test_grayscale_input(self):
        """L mode → 3-channel tensor, shape (3, 28, 28)."""
        img = self._make_image("L")
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 28, 28), f"Expected (3,28,28), got {tensor.shape}"

    def test_rgb_input(self):
        """RGB mode → shape (3, 28, 28), values in [-1, 1]."""
        img = self._make_image("RGB")
        tensor = transform(img)
        assert tensor.shape == (3, 28, 28)
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0

    def test_rgba_input(self):
        """RGBA mode is auto-converted to RGB internally, output (3,28,28)."""
        img = self._make_image("RGBA")
        tensor = transform(img)
        assert tensor.shape == (3, 28, 28)

    def test_large_image_resize(self):
        """A large image (200x200) should be resized to 28x28."""
        img = self._make_image("RGB", size=(200, 200))
        tensor = transform(img)
        assert tensor.shape == (3, 28, 28)

    def test_unsupported_mode(self):
        """P mode (palette) should raise ValueError."""
        img = self._make_image("P")
        with pytest.raises(ValueError, match="Unsupported image mode"):
            transform(img)
