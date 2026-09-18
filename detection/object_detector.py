"""Classical foreground/contour based object detector for the MVP."""

from dataclasses import dataclass

import cv2
import numpy as np

from config.config import DetectionConfig


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    area: float
    label: str = "moving-object"
    confidence: float = 1.0


class MotionObjectDetector:
    """Detect moving objects from a foreground mask using contours."""

    def __init__(self, cfg: DetectionConfig | None = None) -> None:
        self.cfg = cfg or DetectionConfig()

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remove small holes/noise from a binary foreground mask."""
        k = max(3, int(self.cfg.morphology_kernel))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def detect(self, foreground_mask: np.ndarray) -> list[Detection]:
        """Extract candidate moving-object blobs from a foreground mask."""
        mask = self.clean_mask(foreground_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (self.cfg.min_area <= area <= self.cfg.max_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            detections.append(
                Detection(
                    bbox=(x, y, w, h),
                    centroid=(cx, cy),
                    area=area,
                )
            )

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
