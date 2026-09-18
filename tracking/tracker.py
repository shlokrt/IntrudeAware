"""Simple centroid tracker with persistent IDs."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from math import hypot

from config.config import TrackerConfig
from detection.object_detector import Detection


@dataclass(slots=True)
class Track:
    object_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    label: str
    age: int = 1
    disappeared: int = 0
    history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=30))
    prediction_age: int = 0


class CentroidTracker:
    """Nearest-neighbour centroid tracker suitable for an academic MVP."""

    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.next_id = 1
        self.tracks: OrderedDict[int, Track] = OrderedDict()

    def _register(self, detection: Detection) -> Track:
        track = Track(
            object_id=self.next_id,
            bbox=detection.bbox,
            centroid=detection.centroid,
            label=detection.label,
            history=deque([detection.centroid], maxlen=self.cfg.trail_length),
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    def _deregister(self, object_id: int) -> None:
        self.tracks.pop(object_id, None)

    def tracks_list(self) -> list[Track]:
        """Return the currently active tracks without modifying tracker state."""
        return list(self.tracks.values())

    def update(self, detections: list[Detection]) -> list[Track]:
        """Associate current detections with existing tracks."""
        if not detections:
            for object_id in list(self.tracks):
                track = self.tracks[object_id]
                track.disappeared += 1
                if track.disappeared > self.cfg.max_disappeared:
                    self._deregister(object_id)
            return list(self.tracks.values())

        if not self.tracks:
            return [self._register(d) for d in detections]

        track_items = list(self.tracks.items())
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        candidates: list[tuple[float, int, int]] = []
        for t_idx, (_object_id, track) in enumerate(track_items):
            for d_idx, detection in enumerate(detections):
                distance = hypot(
                    track.centroid[0] - detection.centroid[0],
                    track.centroid[1] - detection.centroid[1],
                )
                candidates.append((distance, t_idx, d_idx))

        for distance, t_idx, d_idx in sorted(candidates):
            object_id, track = track_items[t_idx]
            if distance > self.cfg.max_distance:
                break
            if object_id in used_tracks or d_idx in used_detections:
                continue

            detection = detections[d_idx]
            track.bbox = detection.bbox
            track.centroid = detection.centroid
            track.label = detection.label
            track.age += 1
            track.disappeared = 0
            track.prediction_age = 0
            track.history.append(detection.centroid)
            used_tracks.add(object_id)
            used_detections.add(d_idx)

        for object_id, track in track_items:
            if object_id not in used_tracks:
                track.disappeared += 1
                if track.disappeared > self.cfg.max_disappeared:
                    self._deregister(object_id)

        for d_idx, detection in enumerate(detections):
            if d_idx not in used_detections:
                self._register(detection)

        return list(self.tracks.values())

    def predict(self, frame_shape: tuple[int, int] | None = None) -> list[Track]:
        """Advance active tracks between YOLO inference frames.

        A constant-velocity estimate based on the last two trajectory points
        keeps the existing track IDs, labels, zone state, and overlay moving
        on intermediate frames. Tracks are removed after ``max_disappeared``
        prediction frames without a fresh detector observation.
        """
        height = width = None
        if frame_shape is not None:
            height, width = frame_shape

        for object_id in list(self.tracks):
            track = self.tracks[object_id]
            points = list(track.history)

            if len(points) >= 2:
                (x0, y0), (x1, y1) = points[-2], points[-1]
                dx = x1 - x0
                dy = y1 - y0
            else:
                dx = dy = 0

            cx, cy = track.centroid
            new_x = cx + dx
            new_y = cy + dy

            if width is not None:
                new_x = max(0, min(width - 1, new_x))
            if height is not None:
                new_y = max(0, min(height - 1, new_y))

            old_x, old_y, box_w, box_h = track.bbox
            new_left = old_x + dx
            new_top = old_y + dy
            if width is not None:
                new_left = max(0, min(max(0, width - box_w), new_left))
            if height is not None:
                new_top = max(0, min(max(0, height - box_h), new_top))

            track.centroid = (int(new_x), int(new_y))
            track.bbox = (int(new_left), int(new_top), box_w, box_h)
            track.age += 1
            track.prediction_age += 1
            track.history.append(track.centroid)

            if track.prediction_age > self.cfg.max_disappeared:
                self._deregister(object_id)

        return list(self.tracks.values())
