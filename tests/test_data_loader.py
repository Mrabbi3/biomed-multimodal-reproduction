"""
Unit tests for data loading and preprocessing.

Run: pytest tests/test_data_loader.py -v
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import resize_and_pad, preprocess_image, normalize_for_model


class TestResizeAndPad:
    def test_square_image(self):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        result = resize_and_pad(img, (224, 224))
        assert result.size == (224, 224)

    def test_wide_image(self):
        img = Image.new("RGB", (400, 100), color=(0, 255, 0))
        result = resize_and_pad(img, (224, 224))
        assert result.size == (224, 224)

    def test_tall_image(self):
        img = Image.new("RGB", (100, 400), color=(0, 0, 255))
        result = resize_and_pad(img, (224, 224))
        assert result.size == (224, 224)

    def test_grayscale_conversion(self):
        """Paper: grayscale images converted to 3-channel by stacking."""
        gray = Image.new("L", (100, 100), color=128)
        result = resize_and_pad(gray, (224, 224))
        assert result.mode == "RGB"
        assert result.size == (224, 224)

    def test_rgba_conversion(self):
        rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        result = resize_and_pad(rgba, (224, 224))
        assert result.mode == "RGB"

    def test_preserves_aspect_ratio(self):
        """Content should not be distorted."""
        img = Image.new("RGB", (200, 100), color=(255, 255, 0))
        result = resize_and_pad(img, (224, 224))
        arr = np.array(result)
        # Top and bottom should have padding (black)
        assert arr[0, 112, 0] == 0  # top center should be padding


class TestNormalization:
    def test_range(self):
        arr = np.array([[[0, 128, 255]]], dtype=np.uint8)
        norm = normalize_for_model(arr)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0
        assert norm.dtype == np.float32
