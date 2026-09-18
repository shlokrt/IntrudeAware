"""IntrudeAware Streamlit MVP.

Run with:
    streamlit run app.py
"""

from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from analysis.event_detection import EventDetector, Zone
from analysis.statistics import ProcessingStats
from config.config import (
    DetectionConfig,
    INPUT_DIR,
    MotionConfig,
    YOLOConfig,
    OUTPUT_DIR,
    PreprocessConfig,
    TrackerConfig,
    ZoneConfig,
    ensure_directories,
)
from detection.object_detector import MotionObjectDetector
from detection.yolo_detector import YOLOObjectDetector, cuda_available
from motion.background_subtraction import BackgroundSubtractor
from motion.optical_flow import OpticalFlowAnalyzer
from preprocessing.image_preprocessing import preprocess_frame
from tracking.tracker import CentroidTracker
from utils.logger import get_logger
from utils.video_utils import (
    create_writer,
    get_video_metadata,
    open_capture,
    transcode_for_browser,
)
from visualization.drawing import draw_flow, draw_overlay, draw_tracks, draw_zone


LOGGER = get_logger()

st.set_page_config(page_title="IntrudeAware", page_icon="🎥", layout="wide")
ensure_directories()


@st.cache_data(show_spinner=False)
def save_uploaded_video(video_bytes: bytes, filename: str) -> str:
    safe_name = Path(filename).name.replace(" ", "_")
    target = INPUT_DIR / safe_name
    target.write_bytes(video_bytes)
    return str(target)


def format_video_timestamp(seconds: float) -> str:
    """Format a video-relative timestamp as HH:MM:SS.mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def process_video(
    input_path: str,
    output_path: str,
    preprocess_cfg: PreprocessConfig,
    detection_cfg: DetectionConfig,
    tracker_cfg: TrackerConfig,
    motion_cfg: MotionConfig,
    zone_cfg: ZoneConfig,
    show_flow: bool,
    show_edges: bool,
    show_semantic_labels: bool,
    yolo_confidence: float,
    yolo_model: str,
    yolo_device: str,
    yolo_frame_skip: int,
    yolo_adaptive_skip: bool,
    yolo_velocity_alpha: float,
    yolo_refresh_cooldown: int,
) -> tuple[str, ProcessingStats, list[dict], dict[str, float | int]]:
    cap = open_capture(input_path)
    metadata = get_video_metadata(cap)

    background = BackgroundSubtractor()
    detector = MotionObjectDetector(detection_cfg)
    yolo_detector = (
        YOLOObjectDetector(
            YOLOConfig(
                model=yolo_model,
                confidence=yolo_confidence,
                device=yolo_device,
                frame_skip=yolo_frame_skip,
                adaptive_frame_skip=yolo_adaptive_skip,
                velocity_smoothing_alpha=yolo_velocity_alpha,
                refresh_cooldown_frames=yolo_refresh_cooldown,
            )
        )
        if show_semantic_labels
        else None
    )
    tracker = CentroidTracker(tracker_cfg)
    flow = OpticalFlowAnalyzer(
        max_corners=motion_cfg.flow_points,
        quality_level=motion_cfg.flow_quality,
        min_distance=motion_cfg.flow_min_distance,
        vector_scale=motion_cfg.flow_vector_scale,
    )

    writer = None
    stats = ProcessingStats()
    event_detector = EventDetector(Zone(0, 0, 1, 1))
    event_rows: list[dict] = []

    progress = st.progress(0.0, text="Processing video...")
    status = st.empty()
    total_frames = int(metadata["frame_count"])
    fps = float(metadata["fps"])
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            processed = preprocess_frame(frame, preprocess_cfg)
            frame_out = processed.color.copy()
            h, w = processed.gray.shape[:2]

            if writer is None:
                writer = create_writer(output_path, fps, w, h)
                zone = Zone(
                    int(zone_cfg.x1_ratio * w),
                    int(zone_cfg.y1_ratio * h),
                    int(zone_cfg.x2_ratio * w),
                    int(zone_cfg.y2_ratio * h),
                )
                event_detector = EventDetector(zone)

            foreground = background.apply(processed.enhanced)
            if yolo_detector is not None:
                # Intermediate frames are first predicted by the existing
                # tracker. Adaptive scheduling can then immediately trigger a
                # fresh YOLO run when a predicted track moves quickly or enters
                # the restricted zone. The existing tracker and event pipeline
                # remain unchanged.
                zone_bounds = event_detector.zone.x1, event_detector.zone.y1, event_detector.zone.x2, event_detector.zone.y2
                if yolo_detector.should_detect(frame_index, tracker.tracks_list(), zone_bounds):
                    detections = yolo_detector.detect(frame_out)
                    tracks = tracker.update(detections)
                else:
                    detections = []
                    tracks = tracker.predict((h, w))
                    # A prediction may cross into the restricted zone (or
                    # accelerate) on this exact frame, so allow one immediate
                    # semantic refresh without changing downstream components.
                    if yolo_detector.should_detect(frame_index, tracks, zone_bounds):
                        detections = yolo_detector.detect(frame_out)
                        tracks = tracker.update(detections)
            else:
                detections = detector.detect(foreground)
                tracks = tracker.update(detections)
            flow_magnitude = flow.update(processed.enhanced)

            video_timestamp = frame_index / fps if fps > 0 else 0.0
            frame_events = event_detector.update(
                tracks,
                frame_index,
                timestamp_seconds=video_timestamp,
            )

            draw_zone(frame_out, event_detector.zone)
            draw_tracks(frame_out, tracks, event_detector.zone)
            if show_flow:
                draw_flow(frame_out, flow.last_vectors)
            if show_edges:
                edges_bgr = cv2.cvtColor(processed.edges, cv2.COLOR_GRAY2BGR)
                edges_bgr = cv2.resize(edges_bgr, (w, h))
                frame_out = cv2.addWeighted(frame_out, 0.78, edges_bgr, 0.22, 0)
            draw_overlay(frame_out, frame_index, flow_magnitude, len(frame_events))

            active_tracks = len([track for track in tracks if track.disappeared == 0])
            stats.update(
                len(detections),
                active_tracks,
                flow_magnitude,
                len(frame_events),
            )

            for event in frame_events:
                event_rows.append(
                    {
                        "timestamp_seconds": round(event.timestamp_seconds, 3),
                        "timestamp": format_video_timestamp(event.timestamp_seconds),
                        "frame": event.frame_index,
                        "track_id": event.object_id,
                        "status": event.status,
                        "event_type": f"ZONE_{event.status}",
                        "message": event.message,
                    }
                )

            writer.write(frame_out)
            frame_index += 1

            if total_frames > 0:
                progress.progress(
                    min(frame_index / total_frames, 1.0),
                    text=f"Processing frame {frame_index}/{total_frames}",
                )
            else:
                status.write(f"Processing frame {frame_index}")

        progress.progress(1.0, text="Processing complete")
        status.empty()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return output_path, stats, event_rows, metadata


st.title("IntrudeAware")
st.caption("Intelligent Video Surveillance & Scene Analysis — Computer Vision MVP")

with st.sidebar:
    st.header("Processing settings")
    min_area = st.slider("Minimum moving-object area", 100, 10_000, 800, 100)
    max_area = st.slider("Maximum moving-object area", 10_000, 500_000, 300_000, 10_000)
    max_distance = st.slider("Tracker association distance", 20, 200, 80, 5)
    show_flow = st.checkbox("Show optical-flow vectors", value=True)
    show_edges = st.checkbox("Overlay Canny edges", value=False)
    show_semantic_labels = st.checkbox("Use YOLO semantic detection", value=True)

    yolo_model_options = {
        "Nano (yolo26n) — fastest": "yolo26n.pt",
        "Small (yolo26s)": "yolo26s.pt",
        "Medium (yolo26m)": "yolo26m.pt",
        "Large (yolo26l)": "yolo26l.pt",
        "Extra Large (yolo26x) — most compute": "yolo26x.pt",
    }
    yolo_model_label = st.selectbox(
        "YOLO model size",
        options=list(yolo_model_options.keys()),
        index=0,
        disabled=not show_semantic_labels,
        help="Choose the pretrained YOLO26 detection scale. Larger models generally require more compute.",
    )
    yolo_model = yolo_model_options[yolo_model_label]

    device_options = ["CPU"]
    if cuda_available():
        device_options.append("CUDA (GPU)")
    yolo_device_label = st.selectbox(
        "YOLO inference device",
        options=device_options,
        index=0,
        disabled=not show_semantic_labels,
        help="CUDA is shown only when PyTorch reports a CUDA-capable GPU is available.",
    )
    yolo_device = "cuda:0" if yolo_device_label == "CUDA (GPU)" else "cpu"

    yolo_frame_skip = st.slider(
        "Maximum YOLO frame skip", 1, 10, 3, 1,
        disabled=not show_semantic_labels,
        help="Maximum interval between YOLO runs. With adaptive skipping enabled, YOLO runs sooner for fast motion or tracks inside the restricted zone.",
    )

    yolo_adaptive_skip = st.checkbox(
        "Adaptive YOLO frame skipping",
        value=True,
        disabled=not show_semantic_labels,
        help="Run YOLO more often when tracked objects move quickly or are inside the restricted zone. The frame-skip value acts as the maximum interval.",
    )

    yolo_velocity_alpha = st.slider(
        "Velocity smoothing", 0.10, 1.00, 0.35, 0.05,
        disabled=not show_semantic_labels or not yolo_adaptive_skip,
        help="EWMA smoothing for tracked-object speed. Lower values react more slowly and reduce scheduler jitter.",
    )

    yolo_refresh_cooldown = st.slider(
        "YOLO refresh cooldown (frames)", 1, 5, 2, 1,
        disabled=not show_semantic_labels or not yolo_adaptive_skip,
        help="Minimum gap between YOLO runs during adaptive scheduling.",
    )

    yolo_confidence = st.slider(
        "YOLO confidence", 0.10, 0.90, 0.35, 0.05,
        disabled=not show_semantic_labels,
    )

    if show_semantic_labels:
        st.caption(
            f"YOLO: {yolo_model} • device: {yolo_device} • max frame skip: {yolo_frame_skip} • "
            f"adaptive: {'ON' if yolo_adaptive_skip else 'OFF'} • "
            f"smooth α: {yolo_velocity_alpha:.2f} • cooldown: {yolo_refresh_cooldown} frames • "
            "Existing tracker, restricted-zone events, dashboard, and overlay are unchanged."
        )

    st.divider()
    st.write("**Restricted zone**")
    x1 = st.slider("Zone left", 0.0, 0.9, 0.60, 0.01)
    y1 = st.slider("Zone top", 0.0, 0.9, 0.15, 0.01)
    x2 = st.slider("Zone right", 0.1, 1.0, 0.95, 0.01)
    y2 = st.slider("Zone bottom", 0.1, 1.0, 0.90, 0.01)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is None:
    st.info("Upload a video to run the IntrudeAware MVP. A short 10–30 second video is ideal for testing.")
    st.markdown(
        """
### Pipeline

`Video → Preprocessing → YOLO semantic detection → Tracking → Optical flow → Event analysis → Output video`

Background subtraction remains available as the classical-CV fallback when YOLO semantic detection is disabled. The tracker, event engine, and overlay are unchanged.
"""
    )
    st.stop()

input_path = save_uploaded_video(uploaded.getvalue(), uploaded.name)
st.success(f"Loaded: {uploaded.name}")

if st.button("▶ Process video", type="primary", use_container_width=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_path = str(OUTPUT_DIR / f"IntrudeAware_{timestamp}_raw.mp4")
    output_path = str(OUTPUT_DIR / f"IntrudeAware_{timestamp}.mp4")

    preprocess_cfg = PreprocessConfig()
    detection_cfg = DetectionConfig(min_area=min_area, max_area=max_area)
    tracker_cfg = TrackerConfig(max_distance=float(max_distance))
    motion_cfg = MotionConfig()
    zone_cfg = ZoneConfig(x1_ratio=x1, y1_ratio=y1, x2_ratio=x2, y2_ratio=y2)

    try:
        with st.spinner("Running computer-vision pipeline..."):
            _, stats, event_rows, metadata = process_video(
                input_path,
                raw_output_path,
                preprocess_cfg,
                detection_cfg,
                tracker_cfg,
                motion_cfg,
                zone_cfg,
                show_flow,
                show_edges,
                show_semantic_labels,
                yolo_confidence,
                yolo_model,
                yolo_device,
                yolo_frame_skip,
                yolo_adaptive_skip,
                yolo_velocity_alpha,
                yolo_refresh_cooldown,
            )

            transcode_for_browser(raw_output_path, output_path)
            Path(raw_output_path).unlink(missing_ok=True)

        st.subheader("Processed video")
        st.video(output_path)

        events_df = pd.DataFrame(event_rows)
        if events_df.empty:
            entries = exits = unique_tracks = 0
        else:
            entries = int((events_df["status"] == "ENTRY").sum())
            exits = int((events_df["status"] == "EXIT").sum())
            unique_tracks = int(events_df["track_id"].nunique())

        st.subheader("Processing summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Frames", stats.frame_count)
        m2.metric("Max active tracks", stats.max_active_tracks)
        m3.metric("Mean optical flow", f"{stats.mean_flow_magnitude:.2f}")
        m4.metric("Zone entries", entries)
        m5.metric("Zone exits", exits)

        st.caption(f"Unique tracked IDs involved in zone events: {unique_tracks}")

        st.subheader("Restricted-zone event log")
        if events_df.empty:
            st.success("No restricted-zone entry or exit events were detected.")
        else:
            display_df = events_df[
                [
                    "timestamp",
                    "frame",
                    "track_id",
                    "status",
                    "message",
                ]
            ].copy()
            display_df.columns = [
                "Video Time",
                "Frame",
                "Track ID",
                "Status",
                "Event",
            ]

            filter_status = st.selectbox(
                "Filter events",
                ["ALL", "ENTRY", "EXIT"],
                index=0,
            )
            if filter_status != "ALL":
                filtered_df = display_df[display_df["Status"] == filter_status]
            else:
                filtered_df = display_df

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download event CSV",
                events_df.to_csv(index=False).encode("utf-8"),
                file_name="IntrudeAware_events.csv",
                mime="text/csv",
            )

            last_event = events_df.iloc[-1]
            st.info(
                "Latest event: "
                f"{last_event['status']} — Track ID {int(last_event['track_id'])} "
                f"at {format_video_timestamp(float(last_event['timestamp_seconds']))}."
            )

        st.subheader("Processing metadata")
        st.json(metadata)

        st.success(f"Output saved to `{output_path}`")
    except Exception as exc:
        LOGGER.exception("Video processing failed")
        st.error(f"Processing failed: {exc}")
