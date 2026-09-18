"""Central configuration for IntrudeAware."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


@dataclass(slots=True)
class PreprocessConfig:
    target_width: int = 960
    gaussian_kernel: int = 5
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    canny_low: int = 60
    canny_high: int = 150


@dataclass(slots=True)
class DetectionConfig:
    min_area: int = 800
    max_area: int = 300_000
    threshold: int = 200
    morphology_kernel: int = 5


@dataclass(slots=True)
class YOLOConfig:
    model: str = "yolo26n.pt"
    confidence: float = 0.35
    image_size: int = 640
    device: str = "cpu"
    frame_skip: int = 3
    adaptive_frame_skip: bool = True
    medium_speed_px_per_frame: float = 3.0
    high_speed_px_per_frame: float = 8.0
    velocity_smoothing_alpha: float = 0.35
    refresh_cooldown_frames: int = 2


@dataclass(slots=True)
class TrackerConfig:
    max_distance: float = 80.0
    max_disappeared: int = 15
    trail_length: int = 30


@dataclass(slots=True)
class MotionConfig:
    flow_points: int = 120
    flow_quality: float = 0.01
    flow_min_distance: float = 7.0
    flow_vector_scale: float = 1.5


@dataclass(slots=True)
class ZoneConfig:
    x1_ratio: float = 0.60
    y1_ratio: float = 0.15
    x2_ratio: float = 0.95
    y2_ratio: float = 0.90


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
