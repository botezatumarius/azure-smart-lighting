"""Part 1: capture a webcam frame every few seconds and count people locally."""

import argparse
import itertools
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from smart_lighting.camera import CAMERAS, read_frame
from smart_lighting.detector import PersonDetector
from smart_lighting.results import CountLog, plot_counts

log = logging.getLogger(__name__)


def run(
    read: Callable[[], np.ndarray],
    detector: PersonDetector,
    camera: str,
    output: Path,
    interval: float,
    frames: int | None = None,
) -> None:
    """Process one frame per interval until `frames` are done or Ctrl+C is pressed."""
    images = output / "frames"
    images.mkdir(parents=True, exist_ok=True)
    counts = CountLog(output / "counts.csv")
    processed = 0
    next_capture = time.monotonic()
    try:
        for _ in range(frames) if frames else itertools.count():
            time.sleep(max(0.0, next_capture - time.monotonic()))
            # Fixed cadence.
            next_capture = max(next_capture, time.monotonic()) + interval
            try:
                frame = read()
            except ConnectionError as error:
                log.warning("%s; skipping this frame", error)
                continue
            timestamp = datetime.now().astimezone()
            detection = detector.detect(frame)
            cv2.imwrite(str(images / f"{timestamp:%Y%m%d-%H%M%S}.jpg"), detection.annotated)
            counts.append(timestamp, camera, detection.people, detection.inference_ms)
            processed += 1
            log.info("%s: %d people (%.0f ms)", camera, detection.people, detection.inference_ms)
    finally:
        if processed:
            plot_counts(counts.path, output / "people-over-time.png")
            log.info("Saved %d frames, CSV and graph in %s", processed, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=CAMERAS, default="gaggenau")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames")
    parser.add_argument("--frames", type=int, help="stop after N frames (default: until Ctrl+C)")
    parser.add_argument("--output", type=Path, help="results folder (default: runs/<camera>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        run(
            lambda: read_frame(CAMERAS[args.camera]),
            PersonDetector(),
            args.camera,
            args.output or Path("runs") / args.camera,
            args.interval,
            args.frames,
        )
    except KeyboardInterrupt:
        log.info("Stopped by user")


if __name__ == "__main__":
    main()
