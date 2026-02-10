"""
Image preprocessing following Med-PaLM M methodology.

From the paper (Section 4.2):
- Images resized to 224x224x3
- Original aspect ratio preserved with padding
- Grayscale images converted to 3-channel by stacking
"""

import numpy as np
from PIL import Image
from typing import Tuple


def resize_and_pad(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    padding_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Resize image preserving aspect ratio with padding.

    This follows the Med-PaLM M preprocessing:
    "We resized all the images in MultiMedBench to 224x224x3,
     while preserving the original aspect ratio with padding if needed."

    Args:
        image: PIL Image to resize
        target_size: Target (width, height) tuple
        padding_color: RGB color for padding

    Returns:
        Resized and padded PIL Image
    """
    # Convert to RGB if grayscale (paper: "stacking up the same image along channel dimension")
    if image.mode == "L":
        image = Image.merge("RGB", [image, image, image])
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Calculate scaling factor to preserve aspect ratio
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize with high-quality resampling
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    # Create padded canvas
    padded = Image.new("RGB", target_size, padding_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    return padded


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    Args:
        image_path: Path to the image file
        target_size: Target (width, height)

    Returns:
        Numpy array of shape (H, W, 3), values in [0, 255]
    """
    image = Image.open(image_path)
    processed = resize_and_pad(image, target_size)
    return np.array(processed)


def normalize_for_model(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range for model input.

    Args:
        image_array: Numpy array with values in [0, 255]

    Returns:
        Normalized array with values in [0, 1]
    """
    return image_array.astype(np.float32) / 255.0
