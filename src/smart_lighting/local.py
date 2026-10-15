"""Part 1: capture a webcam frame every few seconds and count people locally."""

import argparse
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from smart_lighting.camera import CAMERAS, capture
from smart_lighting.detector import PersonDetector
from smart_lighting.results import COUNT_FIELDS, CsvLog, plot_counts

log = logging.getLogger(__name__)


def run(
    frames: Iterable[tuple[datetime, np.ndarray]],
    detector: PersonDetector,
    camera: str,
    output: Path,
) -> None:
    """Count people in each frame and save the annotated image, CSV row and graph."""
    images = output / "frames"
    images.mkdir(parents=True, exist_ok=True)
    counts = CsvLog(output / "counts.csv", COUNT_FIELDS)
    processed = 0
    try:
        for timestamp, frame in frames:
            detection = detector.detect(frame)
            cv2.imwrite(str(images / f"{timestamp:%Y%m%d-%H%M%S}.jpg"), detection.annotated)
            counts.append(
                timestamp=timestamp,
                camera=camera,
                people=detection.people,
                inference_ms=detection.inference_ms,
            )
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
            capture(CAMERAS[args.camera], args.interval, args.frames),
            PersonDetector(),
            args.camera,
            args.output or Path("runs") / args.camera,
        )
    except KeyboardInterrupt:
        log.info("Stopped by user")


if __name__ == "__main__":
    main()
