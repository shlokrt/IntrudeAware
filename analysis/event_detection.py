"""Rule-based restricted-zone event detection for the IntrudeAware MVP.

The detector emits entry/exit events only when a tracked object changes its
zone state. Brief detection gaps are treated as temporary tracking loss so an
object that remains inside the zone does not generate repeated entries.
"""

from dataclasses import dataclass

from tracking.tracker import Track


@dataclass(slots=True)
class Zone:
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, point: tuple[int, int]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(slots=True)
class Event:
    """A restricted-zone state transition for one tracked object."""

    frame_index: int
    event_type: str  # "ENTRY" or "EXIT"
    object_id: int
    message: str
    timestamp_seconds: float = 0.0

    @property
    def status(self) -> str:
        """Human-readable event status for the dashboard."""
        return self.event_type.upper()


@dataclass(slots=True)
class ZoneState:
    """State retained for a tracked object while it is being observed."""

    inside: bool
    missing_frames: int = 0
    last_entry_frame: int = -1


class EventDetector:
    """Detect restricted-zone entry/exit transitions without duplicates.

    ``missing_grace_frames`` keeps an object's zone state during short
    detection gaps. This prevents repeated entry events when a tracked object
    briefly disappears and then reappears inside the zone.
    """

    def __init__(
        self,
        zone: Zone,
        *,
        missing_grace_frames: int = 30,
        reentry_cooldown_frames: int = 30,
    ) -> None:
        if missing_grace_frames < 0:
            raise ValueError("missing_grace_frames must be >= 0")
        if reentry_cooldown_frames < 0:
            raise ValueError("reentry_cooldown_frames must be >= 0")

        self.zone = zone
        self.missing_grace_frames = missing_grace_frames
        self.reentry_cooldown_frames = reentry_cooldown_frames
        self.states: dict[int, ZoneState] = {}
        self.events: list[Event] = []

        # Backwards-compatible state view used by existing code/debugging.
        self.previous_inside: dict[int, bool] = {}

    def _emit_event(
        self,
        object_id: int,
        frame_index: int,
        event_type: str,
        timestamp_seconds: float,
    ) -> Event:
        status = event_type.upper()
        verb = "entered" if status == "ENTRY" else "exited"
        event = Event(
            frame_index=frame_index,
            event_type=status,
            object_id=object_id,
            message=f"Object {object_id} {verb} the restricted zone.",
            timestamp_seconds=float(timestamp_seconds),
        )
        self.events.append(event)
        return event

    def update(
        self,
        tracks: list[Track],
        frame_index: int,
        timestamp_seconds: float | None = None,
    ) -> list[Event]:
        """Update zone state and return newly generated entry/exit events.

        An object generates:
        - one ENTRY when it transitions outside -> inside;
        - one EXIT when it transitions inside -> outside.

        Brief tracking gaps preserve the previous state for
        ``missing_grace_frames`` frames, avoiding repeated entries.
        """
        frame_events: list[Event] = []
        observed_ids: set[int] = set()
        event_timestamp = float(timestamp_seconds) if timestamp_seconds is not None else 0.0

        for track in tracks:
            object_id = track.object_id

            if track.disappeared > 0:
                state = self.states.get(object_id)
                if state is not None:
                    state.missing_frames += 1
                continue

            observed_ids.add(object_id)
            inside = self.zone.contains(track.centroid)
            state = self.states.get(object_id)

            if state is None:
                # First observation: entering the zone is an ENTRY event.
                state = ZoneState(inside=inside)
                self.states[object_id] = state
                if inside:
                    event = self._emit_event(
                        object_id,
                        frame_index,
                        "ENTRY",
                        event_timestamp,
                    )
                    state.last_entry_frame = frame_index
                    frame_events.append(event)

            else:
                was_inside = state.inside
                state.missing_frames = 0

                if inside and not was_inside:
                    cooldown_ok = (
                        state.last_entry_frame < 0
                        or frame_index - state.last_entry_frame
                        >= self.reentry_cooldown_frames
                    )
                    if cooldown_ok:
                        event = self._emit_event(
                            object_id,
                            frame_index,
                            "ENTRY",
                            event_timestamp,
                        )
                        state.last_entry_frame = frame_index
                        frame_events.append(event)
                        state.inside = True
                    # During a re-entry cooldown, keep the logical zone state
                    # outside. This prevents an EXIT event later for an entry
                    # that was intentionally suppressed by the cooldown.

                elif not inside and was_inside:
                    event = self._emit_event(
                        object_id,
                        frame_index,
                        "EXIT",
                        event_timestamp,
                    )
                    frame_events.append(event)
                    state.inside = False

            self.previous_inside[object_id] = state.inside

        # Remove only tracks that have been missing long enough. This is the
        # key protection against repeated entries from brief detector gaps.
        for object_id in list(self.states):
            if object_id in observed_ids:
                continue

            state = self.states[object_id]
            if state.missing_frames > self.missing_grace_frames:
                self.states.pop(object_id, None)
                self.previous_inside.pop(object_id, None)

        return frame_events
