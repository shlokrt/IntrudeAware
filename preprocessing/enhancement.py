"""Image enhancement helpers."""

import cv2
import numpy as np


def clahe_enhance(gray: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Improve local contrast using CLAHE."""
    if gray.ndim != 2:
        raise ValueError("clahe_enhance expects a grayscale image")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(gray)
