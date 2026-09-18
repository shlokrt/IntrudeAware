"""Edge detection helpers."""

import cv2
import numpy as np


def canny_edges(gray: np.ndarray, low: int = 60, high: int = 150) -> np.ndarray:
    """Return a binary Canny edge image."""
    return cv2.Canny(gray, low, high)
