"""Video I/O utilities."""

from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg


def open_capture(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def get_video_metadata(cap: cv2.VideoCapture) -> dict[str, float | int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0
    return {"width": width, "height": height, "fps": fps, "frame_count": frame_count}





def transcode_for_browser(source_path: str | Path, output_path: str | Path) -> Path:
    """Transcode an OpenCV-generated MP4 into browser-friendly H.264 MP4.

    Browsers commonly reject OpenCV's mp4v/ MPEG-4 Part 2 output even when the
    file extension is .mp4. H.264 + yuv420p + faststart is a much safer format
    for Streamlit's HTML5 video player.
    """
    source = Path(source_path)
    output = Path(output_path)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        details = result.stderr.strip().splitlines()[-10:]
        detail_text = "\n".join(details)
        raise RuntimeError(
            "FFmpeg could not create the browser-compatible output video."
            f"\n{detail_text}"
        )

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("FFmpeg finished but no usable output video was created.")

    return output

import cv2


def create_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create an MP4 writer using the mp4v codec."""

    fourcc_fn = getattr(cv2, "VideoWriter_fourcc")
    fourcc = int(fourcc_fn(*"mp4v"))

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for: {output_path}"
        )

    return writer