"""Acquire one fresh image per request without retaining a video backlog."""

import logging
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np


def capture_frame(source: str, kind: str, timeout: float = 10.0) -> np.ndarray:
    """Retry transient read/decode failures, including a snapshot being rewritten."""
    for attempt in range(3):
        try:
            return _capture_frame(source, kind, timeout)
        except (OSError, ValueError, cv2.error):
            if attempt == 2:
                raise
            logging.warning("Camera read failed; retrying (%d/2).", attempt + 1)
            time.sleep(0.2 * (attempt + 1))
    raise AssertionError("Unreachable")


def _capture_frame(source: str, kind: str, timeout: float) -> np.ndarray:
    """Read a published JPEG endpoint or the first decoded frame of a live stream.

    A new stream connection per sample avoids buffering old frames while inference
    runs. HLS still has provider-side latency, and connection time is measured.
    """
    if kind == "snapshot":
        request = Request(source, headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=timeout) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError("Snapshot exceeds the 20 MB limit.")
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        capture = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(timeout * 1000),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(timeout * 1000),
            ],
        )
        try:
            if not capture.isOpened():
                raise OSError("Cannot open the webcam stream. Check its direct media URL.")
            success, frame = capture.read()
            if not success:
                raise OSError("The webcam opened but did not provide a readable frame.")
        finally:
            capture.release()
    if frame is None or frame.size == 0:
        raise ValueError("The camera returned an invalid image; check availability and media URL.")
    return frame
