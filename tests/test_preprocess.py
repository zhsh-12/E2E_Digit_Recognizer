"""Tests for common/preprocess.py"""

import io
import pytest
from PIL import Image
from common.preprocess import transform


class TestTransform:
    def test_grayscale_l_mode(self, sample_image_l):
        """L mode image should produce a tensor of shape (1, 3, 28, 28)."""
        img = Image.open(sample_image_l)
        assert img.mode == "L"
        tensor = transform(img)
        assert tensor.shape == (1, 3, 28, 28)
        # Values should be in [-1, 1] range
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_rgb_mode(self, sample_image_rgb):
        """RGB image should produce a tensor of shape (1, 3, 28, 28)."""
        img = Image.open(sample_image_rgb)
        assert img.mode == "RGB"
        tensor = transform(img)
        assert tensor.shape == (1, 3, 28, 28)

    def test_rgba_mode(self, sample_image_rgba):
        """RGBA image should be converted to RGB first, then produce (1, 3, 28, 28)."""
        img = Image.open(sample_image_rgba)
        assert img.mode == "RGBA"
        tensor = transform(img)
        assert tensor.shape == (1, 3, 28, 28)

    def test_cmyk_mode_converted(self, sample_image_cmyk):
        """CMYK image mode is converted to RGB by PIL (no error)."""
        img = Image.open(sample_image_cmyk)
        assert img.mode == "CMYK"
        # PIL's convert('RGB') handles CMYK without error
        tensor = transform(img)
        assert tensor.shape == (1, 3, 28, 28)

    def test_resize_large_image(self, sample_image_large):
        """A 100x200 image should be resized to 28x28."""
        img = Image.open(sample_image_large)
        assert img.size == (100, 200)
        tensor = transform(img)
        assert tensor.shape == (1, 3, 28, 28)

    def test_normalization_range(self, sample_image_rgb):
        """After transform, pixel values should be normalized to [-1, 1]."""
        img = Image.open(sample_image_rgb)
        tensor = transform(img)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0
