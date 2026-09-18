from detection.object_detector import Detection
from tracking.tracker import CentroidTracker


def test_tracker_keeps_id_for_nearby_detection():
    tracker = CentroidTracker()
    first = tracker.update([Detection((10, 10, 20, 20), (20, 20), 400)])
    object_id = first[0].object_id

    second = tracker.update([Detection((14, 12, 20, 20), (24, 22), 400)])
    assert second[0].object_id == object_id


def test_tracker_predicts_between_detector_frames():
    from config.config import TrackerConfig

    tracker = CentroidTracker(TrackerConfig(max_distance=80, max_disappeared=3))
    first = tracker.update([Detection((10, 10, 20, 20), (20, 20), 400, label="PERSON")])
    object_id = first[0].object_id
    tracker.update([Detection((20, 10, 20, 20), (30, 20), 400, label="PERSON")])

    predicted = tracker.predict((100, 120))

    assert predicted[0].object_id == object_id
    assert predicted[0].centroid == (40, 20)
    assert predicted[0].label == "PERSON"
    assert predicted[0].disappeared == 0


def test_tracker_prediction_expires_after_configured_gap():
    from config.config import TrackerConfig

    tracker = CentroidTracker(TrackerConfig(max_distance=80, max_disappeared=1))
    tracker.update([Detection((10, 10, 20, 20), (20, 20), 400)])

    assert len(tracker.predict((100, 120))) == 1
    assert len(tracker.predict((100, 120))) == 0
