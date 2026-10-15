"""Capture frames from public webcams (JPEG snapshot, MJPEG or HLS URLs)."""

import itertools
import logging
import time
from collections.abc import Callable, Iterator
from datetime import datetime

import cv2
import numpy as np

# Public fixed cameras.
CAMERAS = {
    "gaggenau": "https://www.gaggenau.de/webcam/image00001.jpg",
}

TIMEOUT_MS = 10_000

log = logging.getLogger(__name__)


def read_frame(url: str) -> np.ndarray:
    """Return the camera's current frame.

    Reconnecting on every read guarantees a fresh image instead of one buffered
    while the previous frame was being processed. At one frame every few seconds
    the reconnection cost is negligible.
    """
    capture = cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS],
    )
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ConnectionError(f"No frame received from {url}")
    return frame


def capture(
    url: str,
    interval: float,
    frames: int | None = None,
    read: Callable[[str], np.ndarray] = read_frame,
) -> Iterator[tuple[datetime, np.ndarray]]:
    """Yield (timestamp, frame) once per interval, forever unless `frames` is given.

    Captures follow a fixed schedule (t = 0, 5, 10 ... s), so the time spent
    processing a frame does not delay the next one. Failed reads are skipped.
    """
    next_capture = time.monotonic()
    for _ in range(frames) if frames else itertools.count():
        time.sleep(max(0.0, next_capture - time.monotonic()))
        next_capture = max(next_capture, time.monotonic()) + interval
        try:
            frame = read(url)
        except ConnectionError as error:
            log.warning("%s; skipping this frame", error)
            continue
        yield datetime.now().astimezone(), frame


def to_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    """Compress a frame for upload to the cloud."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode frame as JPEG")
    return buffer.tobytes()
