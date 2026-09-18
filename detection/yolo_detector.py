"""YOLO-based semantic detector used by IntrudeAware.

The detector returns the same ``Detection`` objects consumed by the existing
centroid tracker, restricted-zone event engine, dashboard, and video overlay.
Ultralytics YOLO performs per-frame semantic detection; the downstream tracker
continues to assign persistent IntrudeAware track IDs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from config.config import YOLOConfig
from detection.object_detector import Detection


# COCO classes that IntrudeAware currently exposes in the overlay.
DEFAULT_LABEL_MAP = {
    "person": "PERSON",
    "bicycle": "BIKE",
    "motorcycle": "BIKE",
    "car": "CAR",
    "bus": "BUS",
    "truck": "TRUCK",
}


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> Any:
    """Load and cache a YOLO model by model name.

    Ultralytics downloads a pretrained weight file automatically when the
    requested model is not available locally.
    """
    from ultralytics import YOLO

    return YOLO(model_name)


def cuda_available() -> bool:
    """Return True when a CUDA-capable PyTorch device is available."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _inside_zone(track: Any, zone_bounds: tuple[int, int, int, int] | None) -> bool:
    """Return whether the track centroid is inside the restricted zone."""
    if zone_bounds is None:
        return False
    x1, y1, x2, y2 = zone_bounds
    cx, cy = track.centroid
    return x1 <= cx <= x2 and y1 <= cy <= y2


class YOLOObjectDetector:
    """Run semantic YOLO detections and control adaptive inference scheduling.

    The detector returns the same ``Detection`` objects consumed by the existing
    centroid tracker, restricted-zone event engine, dashboard, and video overlay.
    Adaptive scheduling uses an exponentially weighted moving average (EWMA) of
    tracked-object speed plus a short minimum refresh gap (cooldown), which
    prevents noisy motion estimates from switching the detector between intervals
    on adjacent frames.
    """

    def __init__(self, cfg: YOLOConfig | None = None) -> None:
        self.cfg = cfg or YOLOConfig()
        if self.cfg.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if not 0.0 < self.cfg.velocity_smoothing_alpha <= 1.0:
            raise ValueError("velocity_smoothing_alpha must be in (0, 1]")
        if self.cfg.refresh_cooldown_frames < 1:
            raise ValueError("refresh_cooldown_frames must be >= 1")
        if self.cfg.medium_speed_px_per_frame < 0:
            raise ValueError("medium_speed_px_per_frame must be >= 0")
        if self.cfg.high_speed_px_per_frame < 0:
            raise ValueError("high_speed_px_per_frame must be >= 0")
        if self.cfg.high_speed_px_per_frame < self.cfg.medium_speed_px_per_frame:
            raise ValueError("high speed threshold must be >= medium threshold")

        self.model = _load_model(self.cfg.model)
        self.label_map = dict(DEFAULT_LABEL_MAP)
        self.last_detection_frame = -1
        self.smoothed_speed = 0.0

    def reset_schedule(self) -> None:
        """Reset scheduler state before processing a new video."""
        self.last_detection_frame = -1
        self.smoothed_speed = 0.0

    @staticmethod
    def _instantaneous_max_speed(tracks: list[Any] | None) -> float:
        """Return the largest current centroid speed across active tracks."""
        if not tracks:
            return 0.0

        max_speed = 0.0
        for track in tracks:
            if getattr(track, "disappeared", 0) > 0:
                continue
            history = list(getattr(track, "history", []))
            if len(history) < 2:
                continue
            (x0, y0), (x1, y1) = history[-2], history[-1]
            speed = float(np.hypot(x1 - x0, y1 - y0))
            max_speed = max(max_speed, speed)
        return max_speed

    def _update_velocity_smoothing(self, tracks: list[Any] | None) -> float:
        """Update and return the EWMA of the maximum tracked-object speed."""
        instantaneous_speed = self._instantaneous_max_speed(tracks)
        alpha = self.cfg.velocity_smoothing_alpha
        self.smoothed_speed = (
            alpha * instantaneous_speed
            + (1.0 - alpha) * self.smoothed_speed
        )
        return self.smoothed_speed

    def _adaptive_interval(
        self,
        tracks: list[Any] | None,
        zone_bounds: tuple[int, int, int, int] | None,
    ) -> int:
        """Return the desired inference interval before applying cooldown."""
        if not self.cfg.adaptive_frame_skip:
            return self.cfg.frame_skip

        if not tracks:
            self._update_velocity_smoothing(None)
            return self.cfg.frame_skip

        smoothed_speed = self._update_velocity_smoothing(tracks)
        any_inside = any(
            getattr(track, "disappeared", 0) == 0
            and _inside_zone(track, zone_bounds)
            for track in tracks
        )

        # Objects inside the zone receive the shortest desired interval so
        # semantic labels and zone transitions stay responsive.
        if any_inside:
            return 1

        if smoothed_speed >= self.cfg.high_speed_px_per_frame:
            return 1
        if smoothed_speed >= self.cfg.medium_speed_px_per_frame:
            return min(2, self.cfg.frame_skip)
        return self.cfg.frame_skip

    def should_detect(
        self,
        frame_index: int,
        tracks: list[Any] | None = None,
        zone_bounds: tuple[int, int, int, int] | None = None,
    ) -> bool:
        """Return whether YOLO should run on the given video frame.

        Frame 0 always runs YOLO. Thereafter the adaptive policy chooses a
        desired interval. The refresh cooldown imposes a short minimum gap
        between actual YOLO runs, preventing noisy speed estimates from causing
        one-frame-on/one-frame-off oscillation.
        """
        if self.last_detection_frame < 0:
            self.last_detection_frame = frame_index
            self._update_velocity_smoothing(tracks)
            return True

        desired_interval = self._adaptive_interval(tracks, zone_bounds)
        cooldown = min(self.cfg.refresh_cooldown_frames, self.cfg.frame_skip)
        effective_interval = max(desired_interval, cooldown)

        due = (frame_index - self.last_detection_frame) >= effective_interval
        if due:
            self.last_detection_frame = frame_index
        return due

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect supported semantic objects in one BGR OpenCV frame."""
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.cfg.confidence,
            "imgsz": self.cfg.image_size,
            "verbose": False,
        }
        if self.cfg.device:
            kwargs["device"] = self.cfg.device

        results = self.model.predict(**kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)

        names = result.names
        detections: list[Detection] = []
        frame_h, frame_w = frame.shape[:2]

        for box, confidence, class_id in zip(xyxy, confs, classes):
            if isinstance(names, dict):
                raw_name = str(names.get(int(class_id), class_id)).lower()
            else:
                raw_name = str(names[int(class_id)]).lower()

            label = self.label_map.get(raw_name)
            if label is None:
                continue

            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(x1 + 1, min(frame_w, x2))
            y2 = max(y1 + 1, min(frame_h, y2))
            width = x2 - x1
            height = y2 - y1
            area = float(width * height)
            centroid = (x1 + width // 2, y1 + height // 2)

            detections.append(
                Detection(
                    bbox=(x1, y1, width, height),
                    centroid=centroid,
                    area=area,
                    label=label,
                    confidence=float(confidence),
                )
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections
