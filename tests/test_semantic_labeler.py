import numpy as np

from detection.object_detector import Detection
from detection.semantic_labeler import SemanticLabeler, SemanticBox


def test_person_semantic_label_is_assigned_from_overlap(monkeypatch):
    labeler = SemanticLabeler(enable_person_hog=False)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = Detection(
        bbox=(40, 20, 40, 80),
        centroid=(60, 60),
        area=2200.0,
    )
    monkeypatch.setattr(
        labeler,
        "detect_persons",
        lambda _frame: [SemanticBox((38, 18, 44, 84))],
    )

    result = labeler.assign_labels(frame, [detection])

    assert result[0].label == "PERSON"


def test_shape_fallback_labels_wide_compact_blob_as_car():
    labeler = SemanticLabeler(enable_person_hog=False)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    detection = Detection(
        bbox=(20, 40, 100, 50),
        centroid=(70, 65),
        area=3000.0,
    )

    result = labeler.assign_labels(frame, [detection])

    assert result[0].label == "CAR"


def test_shape_fallback_labels_sparse_mid_ratio_blob_as_bike():
    labeler = SemanticLabeler(enable_person_hog=False)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    detection = Detection(
        bbox=(30, 30, 60, 60),
        centroid=(60, 60),
        area=1200.0,
    )

    result = labeler.assign_labels(frame, [detection])

    assert result[0].label == "BIKE"
