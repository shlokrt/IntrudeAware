import numpy as np

from config.config import YOLOConfig
from detection.yolo_detector import YOLOObjectDetector


class FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10.2, 20.4, 60.8, 90.6], [80, 30, 140, 70], [5, 5, 20, 20]])
        self.conf = FakeTensor([0.91, 0.83, 0.95])
        self.cls = FakeTensor([0, 2, 15])

    def __len__(self):
        return 3


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "person", 2: "car", 15: "cat"}


class FakeModel:
    def predict(self, **kwargs):
        assert kwargs["conf"] == 0.4
        assert kwargs["imgsz"] == 640
        assert kwargs["verbose"] is False
        assert kwargs["device"] == "cpu"
        return [FakeResult()]


def test_yolo_detector_maps_supported_classes(monkeypatch):
    monkeypatch.setattr(
        "detection.yolo_detector._load_model",
        lambda _model_name: FakeModel(),
    )
    detector = YOLOObjectDetector(YOLOConfig(confidence=0.4, device="cpu"))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert [d.label for d in detections] == ["PERSON", "CAR"]
    assert detections[0].confidence == 0.91
    assert detections[0].bbox == (10, 20, 51, 71)


def test_yolo_detector_filters_unsupported_classes(monkeypatch):
    monkeypatch.setattr(
        "detection.yolo_detector._load_model",
        lambda _model_name: FakeModel(),
    )
    detector = YOLOObjectDetector(YOLOConfig(confidence=0.4, device="cpu"))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert all(d.label in {"PERSON", "BIKE", "CAR", "BUS", "TRUCK"} for d in detections)
    assert len(detections) == 2


class DeviceCheckingFakeModel:
    def __init__(self):
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        return []


def test_yolo_detector_forwards_selected_device(monkeypatch):
    fake = DeviceCheckingFakeModel()
    monkeypatch.setattr(
        "detection.yolo_detector._load_model",
        lambda _model_name: fake,
    )
    detector = YOLOObjectDetector(
        YOLOConfig(model="yolo26s.pt", confidence=0.35, device="cuda:0")
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detector.detect(frame)
    assert fake.last_kwargs["device"] == "cuda:0"


class FakeTrack:
    def __init__(self, history, centroid, disappeared=0):
        self.history = history
        self.centroid = centroid
        self.disappeared = disappeared


def test_yolo_frame_skip_schedule(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(YOLOConfig(model="yolo26n.pt", frame_skip=3, adaptive_frame_skip=False))
    assert detector.should_detect(0) is True
    assert detector.should_detect(1) is False
    assert detector.should_detect(2) is False
    assert detector.should_detect(3) is True


def test_yolo_frame_skip_must_be_positive(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    try:
        YOLOObjectDetector(YOLOConfig(frame_skip=0))
        assert False, "Expected ValueError for frame_skip=0"
    except ValueError as exc:
        assert "frame_skip" in str(exc)


def test_adaptive_skip_reduces_interval_for_fast_motion(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(
        YOLOConfig(
            frame_skip=5,
            adaptive_frame_skip=True,
            medium_speed_px_per_frame=3.0,
            high_speed_px_per_frame=8.0,
            refresh_cooldown_frames=2,
        )
    )
    track = FakeTrack([(10, 10), (22, 10)], (22, 10))
    zone = (80, 80, 100, 100)
    assert detector.should_detect(0, [track], zone) is True
    assert detector.should_detect(1, [track], zone) is False
    assert detector.should_detect(2, [track], zone) is True


def test_adaptive_skip_runs_inside_zone_after_cooldown(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(YOLOConfig(frame_skip=5, adaptive_frame_skip=True, refresh_cooldown_frames=2))
    track = FakeTrack([(50, 50), (51, 50)], (51, 50))
    zone = (40, 40, 70, 70)
    assert detector.should_detect(0, [track], zone) is True
    assert detector.should_detect(1, [track], zone) is False
    assert detector.should_detect(2, [track], zone) is True


def test_adaptive_skip_keeps_max_interval_when_scene_is_calm(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(YOLOConfig(frame_skip=3, adaptive_frame_skip=True, refresh_cooldown_frames=2))
    track = FakeTrack([(10, 10), (11, 10)], (11, 10))
    zone = (80, 80, 100, 100)
    assert detector.should_detect(0, [track], zone) is True
    assert detector.should_detect(1, [track], zone) is False
    assert detector.should_detect(2, [track], zone) is False
    assert detector.should_detect(3, [track], zone) is True


def test_adaptive_skip_can_be_disabled(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(YOLOConfig(frame_skip=3, adaptive_frame_skip=False))
    track = FakeTrack([(10, 10), (30, 10)], (30, 10))
    zone = (20, 0, 50, 50)
    assert detector.should_detect(0, [track], zone) is True
    assert detector.should_detect(1, [track], zone) is False
    assert detector.should_detect(2, [track], zone) is False
    assert detector.should_detect(3, [track], zone) is True


def test_velocity_smoothing_reduces_spike(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(
        YOLOConfig(frame_skip=5, adaptive_frame_skip=True, velocity_smoothing_alpha=0.25)
    )
    calm = FakeTrack([(0, 0), (1, 0)], (1, 0))
    fast = FakeTrack([(0, 0), (20, 0)], (20, 0))
    detector.should_detect(0, [calm], None)
    detector.should_detect(5, [fast], None)
    assert 0.0 < detector.smoothed_speed < 20.0


def test_velocity_smoothing_alpha_one_tracks_latest_speed(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(
        YOLOConfig(frame_skip=5, adaptive_frame_skip=True, velocity_smoothing_alpha=1.0)
    )
    fast = FakeTrack([(0, 0), (9, 0)], (9, 0))
    detector.should_detect(0, [fast], None)
    assert detector.smoothed_speed == 9.0


def test_refresh_cooldown_prevents_one_frame_oscillation(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(
        YOLOConfig(frame_skip=5, adaptive_frame_skip=True, refresh_cooldown_frames=2)
    )
    fast = FakeTrack([(0, 0), (20, 0)], (20, 0))
    assert detector.should_detect(0, [fast], None) is True
    assert detector.should_detect(1, [fast], None) is False
    assert detector.should_detect(2, [fast], None) is True
    assert detector.should_detect(3, [fast], None) is False


def test_cooldown_does_not_override_frame_skip_one(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    detector = YOLOObjectDetector(
        YOLOConfig(frame_skip=1, adaptive_frame_skip=True, refresh_cooldown_frames=4)
    )
    fast = FakeTrack([(0, 0), (20, 0)], (20, 0))
    assert detector.should_detect(0, [fast], None) is True
    assert detector.should_detect(1, [fast], None) is True
    assert detector.should_detect(2, [fast], None) is True


def test_invalid_velocity_smoothing_alpha_is_rejected(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    try:
        YOLOObjectDetector(YOLOConfig(velocity_smoothing_alpha=0.0))
        assert False, "Expected ValueError for velocity_smoothing_alpha=0"
    except ValueError as exc:
        assert "velocity_smoothing_alpha" in str(exc)


def test_invalid_refresh_cooldown_is_rejected(monkeypatch):
    monkeypatch.setattr("detection.yolo_detector._load_model", lambda _model_name: FakeModel())
    try:
        YOLOObjectDetector(YOLOConfig(refresh_cooldown_frames=0))
        assert False, "Expected ValueError for refresh_cooldown_frames=0"
    except ValueError as exc:
        assert "refresh_cooldown_frames" in str(exc)
