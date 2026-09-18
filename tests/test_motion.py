import numpy as np

from motion.background_subtraction import BackgroundSubtractor
from motion.optical_flow import OpticalFlowAnalyzer


def test_background_subtractor_returns_mask():
    subtractor = BackgroundSubtractor()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = subtractor.apply(frame)
    assert mask.shape == frame.shape[:2]


def test_optical_flow_initial_frame_is_zero():
    flow = OpticalFlowAnalyzer()
    gray = np.zeros((80, 100), dtype=np.uint8)
    magnitude = flow.update(gray)
    assert magnitude == 0.0


def test_optical_flow_handles_tracked_points():
    flow = OpticalFlowAnalyzer(max_corners=20)
    frame1 = np.zeros((100, 120), dtype=np.uint8)
    frame2 = np.zeros((100, 120), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(frame1, (30, 30), (60, 60), 255, -1)
    cv2.rectangle(frame2, (34, 30), (64, 60), 255, -1)

    flow.update(frame1)
    magnitude = flow.update(frame2)

    assert isinstance(magnitude, float)
    assert magnitude >= 0.0
