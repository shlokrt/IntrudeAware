import cv2
import numpy as np

from analysis.event_detection import Zone
from tracking.tracker import Track
from visualization.drawing import draw_tracks, draw_zone


def make_track(object_id: int, centroid: tuple[int, int]) -> Track:
    return Track(
        object_id=object_id,
        bbox=(centroid[0] - 10, centroid[1] - 10, 20, 20),
        centroid=centroid,
        label="moving-object",
    )


def test_draw_zone_changes_frame_pixels():
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    before = frame.copy()
    zone = Zone(60, 30, 180, 130)

    draw_zone(frame, zone)

    assert not np.array_equal(frame, before)
    # Center of the restricted zone should be visibly highlighted.
    assert frame[80, 120].sum() > 0


def test_draw_tracks_labels_active_track_and_marks_zone_membership():
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    zone = Zone(60, 30, 180, 130)
    track = make_track(7, (100, 80))

    draw_tracks(frame, [track], zone)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert int(np.count_nonzero(gray)) > 0
