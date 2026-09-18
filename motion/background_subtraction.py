"""Background subtraction using OpenCV MOG2."""

import cv2
import numpy as np


class BackgroundSubtractor:
    def __init__(self, history: int = 500, var_threshold: float = 16.0, detect_shadows: bool = True) -> None:
        self.model = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def apply(self, frame: np.ndarray, learning_rate: float = -1.0) -> np.ndarray:
        """Return a binary-ish foreground mask."""
        mask = self.model.apply(frame, learningRate=learning_rate)
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return binary
