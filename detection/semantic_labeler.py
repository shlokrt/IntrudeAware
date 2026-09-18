"""Lightweight semantic labeling for moving-object detections.

The MVP remains classical computer vision: HOG is used to recognize people,
while simple geometry/extent cues provide CAR/BIKE/VEHICLE labels for other
foreground blobs. This keeps the project dependency-light and explainable.
For production-grade semantic detection, this module can later be replaced by
a learned detector without changing the tracking or visualization layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

import cv2
import numpy as np

from detection.object_detector import Detection


@dataclass(slots=True)
class SemanticBox:
    """A semantic person detection produced by HOG."""

    bbox: tuple[int, int, int, int]
    label: str = "PERSON"

    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


class SemanticLabeler:
    """Assign lightweight semantic labels to moving-object detections."""

    PERSON_IOU_THRESHOLD = 0.12
    PERSON_CENTER_DISTANCE_FACTOR = 0.55

    def __init__(self, enable_person_hog: bool = True) -> None:
        self.enable_person_hog = enable_person_hog
        self.hog: cv2.HOGDescriptor | None = None
        if enable_person_hog:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @staticmethod
    def _iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection == 0:
            return 0.0

        union = aw * ah + bw * bh - intersection
        return float(intersection / union) if union else 0.0

    @staticmethod
    def _center_distance(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ac = (ax + aw // 2, ay + ah // 2)
        bc = (bx + bw // 2, by + bh // 2)
        return hypot(ac[0] - bc[0], ac[1] - bc[1])

    def detect_persons(self, frame: np.ndarray) -> list[SemanticBox]:
        """Detect people with OpenCV's built-in HOG person detector."""
        if not self.enable_person_hog or self.hog is None:
            return []

        # HOG can be expensive on very large frames, so keep inference bounded.
        max_width = 960
        image = frame
        scale = 1.0
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            new_size = (max_width, int(frame.shape[0] * scale))
            image = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        rects, _weights = self.hog.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        boxes: list[SemanticBox] = []
        for x, y, w, h in rects:
            if scale != 1.0:
                x = int(round(x / scale))
                y = int(round(y / scale))
                w = int(round(w / scale))
                h = int(round(h / scale))
            boxes.append(SemanticBox((int(x), int(y), int(w), int(h))))
        return boxes

    def _classify_non_person(self, detection: Detection) -> str:
        """Use shape/extent cues for a transparent classical fallback label."""
        x, y, w, h = detection.bbox
        del x, y  # geometry only; kept for readability of the bbox tuple
        aspect = w / max(h, 1)
        extent = detection.area / max(w * h, 1)

        # Broad, explainable heuristics. These are not learned predictions.
        if aspect >= 1.55 and extent >= 0.42:
            return "CAR"
        if 0.65 <= aspect <= 1.75 and extent < 0.42:
            return "BIKE"
        if aspect > 2.0:
            return "BUS/TRUCK"
        return "VEHICLE"

    def assign_labels(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
    ) -> list[Detection]:
        """Return detections with semantic labels assigned."""
        detections_list = list(detections)
        person_boxes = self.detect_persons(frame)

        labeled: list[Detection] = []
        for detection in detections_list:
            label = None
            for person in person_boxes:
                iou = self._iou(detection.bbox, person.bbox)
                distance = self._center_distance(detection.bbox, person.bbox)
                x, y, w, h = person.bbox
                person_diag = max(1.0, hypot(w, h))
                if iou >= self.PERSON_IOU_THRESHOLD or distance <= self.PERSON_CENTER_DISTANCE_FACTOR * person_diag:
                    label = "PERSON"
                    break

            if label is None:
                label = self._classify_non_person(detection)

            labeled.append(
                Detection(
                    bbox=detection.bbox,
                    centroid=detection.centroid,
                    area=detection.area,
                    label=label,
                    confidence=detection.confidence,
                )
            )

        return labeled
