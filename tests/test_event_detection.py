from analysis.event_detection import EventDetector, Zone
from tracking.tracker import Track


def make_track(object_id: int, centroid: tuple[int, int], disappeared: int = 0) -> Track:
    return Track(
        object_id=object_id,
        bbox=(centroid[0] - 5, centroid[1] - 5, 10, 10),
        centroid=centroid,
        label="moving-object",
        disappeared=disappeared,
    )


def test_inside_object_emits_only_one_entry_event():
    detector = EventDetector(Zone(0, 0, 100, 100))
    track = make_track(1, (50, 50))

    assert len(detector.update([track], 0, 0.0)) == 1
    assert len(detector.update([track], 1, 0.04)) == 0
    assert len(detector.update([track], 2, 0.08)) == 0
    assert len(detector.update([track], 3, 0.12)) == 0

    assert len(detector.events) == 1
    assert detector.events[0].status == "ENTRY"
    assert detector.events[0].object_id == 1
    assert detector.events[0].timestamp_seconds == 0.0


def test_temporary_detection_gap_does_not_repeat_entry():
    detector = EventDetector(Zone(0, 0, 100, 100), missing_grace_frames=3)
    inside = make_track(1, (50, 50))
    missing = make_track(1, (50, 50), disappeared=1)

    assert len(detector.update([inside], 0, 0.0)) == 1
    assert len(detector.update([missing], 1, 0.04)) == 0
    assert len(detector.update([missing], 2, 0.08)) == 0
    assert len(detector.update([inside], 3, 0.12)) == 0
    assert len(detector.events) == 1


def test_exit_event_is_generated_on_inside_to_outside_transition():
    detector = EventDetector(Zone(0, 0, 100, 100))
    inside = make_track(7, (50, 50))
    outside = make_track(7, (150, 150))

    assert len(detector.update([inside], 10, 0.4)) == 1
    events = detector.update([outside], 20, 0.8)

    assert len(events) == 1
    assert events[0].status == "EXIT"
    assert events[0].object_id == 7
    assert events[0].frame_index == 20
    assert events[0].timestamp_seconds == 0.8


def test_leave_and_reenter_generates_new_entry_after_cooldown():
    detector = EventDetector(Zone(0, 0, 100, 100), reentry_cooldown_frames=5)
    inside = make_track(1, (50, 50))
    outside = make_track(1, (150, 150))

    assert len(detector.update([inside], 0)) == 1
    assert len(detector.update([inside], 1)) == 0
    assert len(detector.update([outside], 2)) == 1  # EXIT
    assert len(detector.update([inside], 3)) == 0  # cooldown still active
    assert len(detector.update([outside], 4)) == 0
    assert len(detector.update([inside], 6)) == 1  # new ENTRY

    assert [event.status for event in detector.events] == ["ENTRY", "EXIT", "ENTRY"]
