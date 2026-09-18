"""Sparse Lucas–Kanade optical-flow analysis."""

import cv2
import numpy as np


class OpticalFlowAnalyzer:
    def __init__(
        self,
        max_corners: int = 120,
        quality_level: float = 0.01,
        min_distance: float = 7.0,
        vector_scale: float = 1.5,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.vector_scale = vector_scale
        self.prev_gray: np.ndarray | None = None
        self.last_vectors: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        self.mean_magnitude = 0.0

    def reset(self) -> None:
        self.prev_gray = None
        self.last_vectors.clear()
        self.mean_magnitude = 0.0

    def update(self, gray: np.ndarray) -> float:
        """Estimate sparse optical flow against the previous frame."""
        self.last_vectors = []
        self.mean_magnitude = 0.0

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return 0.0

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if prev_pts is None:
            self.prev_gray = gray.copy()
            return 0.0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
        if curr_pts is None or status is None:
            self.prev_gray = gray.copy()
            return 0.0

        prev_good = prev_pts[status.flatten() == 1].reshape(-1, 2)
        curr_good = curr_pts[status.flatten() == 1].reshape(-1, 2)
        magnitudes: list[float] = []

        # OpenCV returns point arrays shaped like (N, 1, 2). After filtering,
        # each point can therefore still be shaped (1, 2). Flattening to (2,)
        # before scalar conversion avoids the NumPy error:
        # "only length-1 arrays can be converted to Python scalars".
        for p0, p1 in zip(prev_good, curr_good):
            x0 = int(round(float(p0[0])))
            y0 = int(round(float(p0[1])))
            x1 = int(round(float(p1[0])))
            y1 = int(round(float(p1[1])))
            magnitude = float(np.hypot(x1 - x0, y1 - y0))
            if magnitude < 0.5:
                continue
            self.last_vectors.append(((x0, y0), (x1, y1), magnitude))
            magnitudes.append(magnitude)

        if magnitudes:
            self.mean_magnitude = float(np.mean(magnitudes))

        self.prev_gray = gray.copy()
        return self.mean_magnitude
