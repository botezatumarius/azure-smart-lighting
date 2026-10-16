"""Send captured frames to a cloud target and log what happened to each one.

A target is any object with `submit(camera, timestamp, jpeg) -> dict` that raises
ConnectionError on failure: the Azure VM (Part 2) or Blob Storage (Part 3).
"""

import logging
import time
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

import numpy as np

from smart_lighting.camera import to_jpeg
from smart_lighting.results import COUNT_FIELDS, CsvLog

SENT_FIELDS = (*COUNT_FIELDS, "response_ms", "bytes_sent")

log = logging.getLogger(__name__)


class Target(Protocol):
    def submit(self, camera: str, timestamp: datetime, jpeg: bytes) -> dict: ...


def stream(
    frames: Iterable[tuple[datetime, np.ndarray]], target: Target, camera: str, sent: CsvLog
) -> int:
    """Submit each frame; log the result, round-trip time and upload size. Returns frames sent."""
    count = 0
    for timestamp, frame in frames:
        jpeg = to_jpeg(frame)
        started = time.perf_counter()
        try:
            result = target.submit(camera, timestamp, jpeg)
        except ConnectionError as error:
            log.warning("%s: %s", camera, error)
            continue
        response_ms = (time.perf_counter() - started) * 1000
        sent.append(
            timestamp=timestamp,
            camera=camera,
            response_ms=response_ms,
            bytes_sent=len(jpeg),
            **result,
        )
        count += 1
        log.info("%s: sent %d kB in %.0f ms %s", camera, len(jpeg) // 1000, response_ms, result)
    return count
