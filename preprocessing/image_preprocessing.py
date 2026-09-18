"""Core image preprocessing pipeline."""

from dataclasses import dataclass

import cv2
import numpy as np

from config.config import PreprocessConfig
from preprocessing.edge_detection import canny_edges
from preprocessing.enhancement import clahe_enhance


@dataclass(slots=True)
class PreprocessedFrame:
    color: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    enhanced: np.ndarray
    edges: np.ndarray


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Resize a frame to a target width while preserving aspect ratio."""
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame.copy()
    scale = target_width / float(width)
    target_height = max(1, int(height * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def preprocess_frame(frame: np.ndarray, cfg: PreprocessConfig | None = None) -> PreprocessedFrame:
    """Run the preprocessing stages used by the MVP."""
    cfg = cfg or PreprocessConfig()
    color = resize_keep_aspect(frame, cfg.target_width)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    k = cfg.gaussian_kernel
    if k % 2 == 0:
        raise ValueError("gaussian_kernel must be odd")
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    enhanced = clahe_enhance(blurred, cfg.clahe_clip_limit, cfg.clahe_grid_size)
    edges = canny_edges(enhanced, cfg.canny_low, cfg.canny_high)

    return PreprocessedFrame(
        color=color,
        gray=gray,
        blurred=blurred,
        enhanced=enhanced,
        edges=edges,
    )
