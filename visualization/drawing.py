"""OpenCV drawing functions for annotated frames."""

from collections.abc import Iterable

import cv2
import numpy as np

from analysis.event_detection import Zone
from tracking.tracker import Track


# BGR colors used only for video annotations.
ZONE_COLOR = (40, 40, 220)          # red
TRACK_COLOR = (60, 210, 60)         # green
TRACK_INSIDE_COLOR = (40, 210, 255) # yellow/orange
LABEL_BG = (18, 18, 18)
TEXT_COLOR = (245, 245, 245)
TRAIL_COLOR = (255, 170, 0)
CENTROID_COLOR = (255, 200, 0)


def draw_zone(frame: np.ndarray, zone: Zone) -> None:
    """Highlight the restricted zone with a translucent fill and label."""
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (zone.x1, zone.y1),
        (zone.x2, zone.y2),
        ZONE_COLOR,
        thickness=-1,
    )
    # Keep the scene visible while clearly showing the protected region.
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, dst=frame)

    cv2.rectangle(
        frame,
        (zone.x1, zone.y1),
        (zone.x2, zone.y2),
        ZONE_COLOR,
        3,
    )

    label = "RESTRICTED ZONE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_x = max(4, zone.x1 + 8)
    label_y = max(text_h + baseline + 8, zone.y1 + text_h + baseline + 8)

    # Clamp the label so it stays inside the frame.
    label_x = min(label_x, max(4, frame.shape[1] - text_w - 12))
    label_y = min(label_y, max(text_h + baseline + 4, frame.shape[0] - 4))

    cv2.rectangle(
        frame,
        (label_x - 6, label_y - text_h - baseline - 6),
        (label_x + text_w + 6, label_y + baseline + 2),
        LABEL_BG,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    zone: Zone | None = None,
) -> None:
    """Draw active tracks, their IDs, centroids, and recent trajectories.

    Tracks with ``disappeared > 0`` are not drawn because they are not currently
    active in the processed frame. When a zone is supplied, tracks inside the
    restricted zone receive a distinct highlight color.
    """
    for track in tracks:
        if track.disappeared > 0:
            continue

        x, y, w, h = track.bbox
        inside_zone = zone is not None and zone.contains(track.centroid)
        track_color = TRACK_INSIDE_COLOR if inside_zone else TRACK_COLOR

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            track_color,
            3,
        )
        cv2.circle(frame, track.centroid, 5, CENTROID_COLOR, -1)

        # Draw the recent trajectory behind/around the current bounding box.
        points = list(track.history)
        for idx in range(1, len(points)):
            cv2.line(
                frame,
                points[idx - 1],
                points[idx],
                TRAIL_COLOR,
                2,
            )

        semantic_label = (track.label or "MOVING-OBJECT").upper()
        label = f"ID {track.object_id} | {semantic_label}"
        if inside_zone:
            label += " | IN ZONE"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        text_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness,
        )

        # Prefer a label above the box, but keep it visible on-screen.
        label_x = max(2, x)
        label_y = y - 8
        if label_y - text_h - baseline < 2:
            label_y = min(
                frame.shape[0] - baseline - 2,
                y + text_h + baseline + 8,
            )

        label_x = min(label_x, max(2, frame.shape[1] - text_w - 8))

        cv2.rectangle(
            frame,
            (label_x - 4, label_y - text_h - baseline - 4),
            (label_x + text_w + 4, label_y + baseline + 2),
            LABEL_BG,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            font,
            font_scale,
            track_color,
            text_thickness,
            cv2.LINE_AA,
        )


def draw_flow(
    frame: np.ndarray,
    vectors: Iterable[tuple[tuple[int, int], tuple[int, int], float]],
) -> None:
    for (x0, y0), (x1, y1), _magnitude in vectors:
        cv2.arrowedLine(
            frame,
            (x0, y0),
            (x1, y1),
            (255, 80, 80),
            1,
            tipLength=0.25,
        )


def draw_overlay(
    frame: np.ndarray,
    frame_index: int,
    flow_magnitude: float,
    zone_events: int,
) -> None:
    cv2.rectangle(frame, (8, 8), (285, 96), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Frame: {frame_index}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )
    cv2.putText(
        frame,
        f"Optical flow: {flow_magnitude:.2f}",
        (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )
    cv2.putText(
        frame,
        f"New events: {zone_events}",
        (18, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )
