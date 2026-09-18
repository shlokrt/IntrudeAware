"""Runtime statistics collection."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ProcessingStats:
    frame_count: int = 0
    detected_count_total: int = 0
    max_active_tracks: int = 0
    mean_flow_magnitude: float = 0.0
    flow_samples: int = 0
    zone_events: int = 0
    event_frames: list[int] = field(default_factory=list)

    def update(self, detection_count: int, active_tracks: int, flow_magnitude: float, event_count: int) -> None:
        self.frame_count += 1
        self.detected_count_total += detection_count
        self.max_active_tracks = max(self.max_active_tracks, active_tracks)
        if flow_magnitude > 0:
            self.mean_flow_magnitude = (
                self.mean_flow_magnitude * self.flow_samples + flow_magnitude
            ) / (self.flow_samples + 1)
            self.flow_samples += 1
        self.zone_events += event_count
