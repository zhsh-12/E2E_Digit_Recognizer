import numpy as np
from PIL import Image


def transform(img: Image.Image) -> np.ndarray:
    """
    Preprocess image for ONNX model inference.
    Replaces torchvision transforms with PIL + numpy to avoid torchvision dependency.
    """
    # Convert to RGB (handles L, RGBA, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array (H, W, C) and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Normalize to [-1, 1]: (pixel - 0.5) / 0.5
    img_array = (img_array - 0.5) / 0.5

    # Transpose from (H, W, C) to (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension -> (1, C, H, W)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
