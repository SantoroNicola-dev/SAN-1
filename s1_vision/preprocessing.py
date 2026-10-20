"""
Image Preprocessing Module for S1stem Vision Pipeline.
Provides standardized image transformation, normalization, and tensor conversion for ONNX Runtime.
"""

from typing import Union, Tuple, Optional
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image

# Standard ImageNet Mean and Standard Deviation
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class ImagePreprocessor:
    """
    Handles robust image ingestion and transformation for ONNX Vision models.
    Supports file paths, PIL Images, raw bytes, base64 strings, and numpy arrays.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.mean = mean if mean is not None else DEFAULT_MEAN
        self.std = std if std is not None else DEFAULT_STD
        self.normalize = normalize

    def load_image(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Image.Image:
        """Loads and converts any supported image representation into an RGB PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if path_str.startswith("data:image/") and ";base64," in path_str:
                _, b64_data = path_str.split(";base64,", 1)
                img_bytes = base64.b64decode(b64_data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return Image.open(path_str).convert("RGB")

        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:  # Grayscale
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                if image_input.shape[0] == 3:  # CHW -> HWC
                    image_input = np.transpose(image_input, (1, 2, 0))
                if image_input.dtype != np.uint8:
                    if image_input.max() <= 1.0:
                        image_input = (image_input * 255.0).astype(np.uint8)
                    else:
                        image_input = image_input.astype(np.uint8)
                return Image.fromarray(image_input).convert("RGB")

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def preprocess(
        self, 
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Processes an image into a normalized NCHW numpy array [1, 3, H, W] of float32.
        
        Steps:
        1. Resize maintaining aspect ratio with bicubic interpolation.
        2. Center crop to (target_size).
        3. Convert pixels to [0.0, 1.0].
        4. Normalize with (x - mean) / std.
        5. Transpose HWC -> CHW and add batch dimension [1, 3, H, W].
        """
        pil_img = self.load_image(image_input)

        # High-quality resize & center crop
        w, h = pil_img.size
        tw, th = self.target_size
        scale = max(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = pil_img.resize((nw, nh), resample=Image.Resampling.BICUBIC)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        cropped = resized.crop((left, top, left + tw, top + th))

        # Convert to numpy array [H, W, 3] in float32 [0.0, 1.0]
        arr = np.array(cropped, dtype=np.float32) / 255.0

        # Transpose HWC -> CHW [3, H, W]
        chw = np.transpose(arr, (2, 0, 1))

        # Apply mean/std normalization
        if self.normalize:
            chw = (chw - self.mean) / self.std

        # Add batch dimension [1, 3, H, W]
        nchw = np.expand_dims(chw, axis=0).astype(np.float32)
        return nchw

    def preprocess_batch(
        self,
        image_inputs: list[Union[str, Path, bytes, Image.Image, np.ndarray]]
    ) -> np.ndarray:
        """
        Processes a list of image inputs into a single normalized tensor [B, 3, H, W].
        """
        if not image_inputs:
            tw, th = self.target_size
            return np.empty((0, 3, th, tw), dtype=np.float32)

        batch_tensors = [self.preprocess(img)[0] for img in image_inputs]
        return np.stack(batch_tensors, axis=0).astype(np.float32)

