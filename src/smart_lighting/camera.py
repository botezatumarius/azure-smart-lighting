"""Grab single frames from public webcams (JPEG snapshot, MJPEG or HLS URLs)."""

import cv2
import numpy as np

# Public fixed cameras.
CAMERAS = {
    "gaggenau": "https://www.gaggenau.de/webcam/image00001.jpg",
}

TIMEOUT_MS = 10_000


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
