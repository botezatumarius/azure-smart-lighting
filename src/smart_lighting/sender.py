"""Part 2: capture frames locally and send them to the Azure VM for people counting."""

import argparse
import logging
import os
import time
from pathlib import Path

import requests

from smart_lighting.camera import CAMERAS, capture, to_jpeg
from smart_lighting.results import COUNT_FIELDS, CsvLog, plot_counts

SENT_FIELDS = (*COUNT_FIELDS, "response_ms", "bytes_sent")

log = logging.getLogger(__name__)


def send(frames, server: str, token: str, camera: str, output: Path) -> None:
    """POST each frame to the VM and log its answer and the round-trip time."""
    output.mkdir(parents=True, exist_ok=True)
    sent = CsvLog(output / "sent.csv", SENT_FIELDS)
    session = requests.Session()
    session.headers.update({"X-API-Key": token, "Content-Type": "image/jpeg"})
    processed = 0
    try:
        for timestamp, frame in frames:
            jpeg = to_jpeg(frame)
            started = time.perf_counter()
            try:
                response = session.post(
                    f"{server}/frames", params={"camera": camera}, data=jpeg, timeout=60
                )
                response.raise_for_status()
            except requests.RequestException as error:
                log.warning("Upload failed: %s", error)
                continue
            response_ms = (time.perf_counter() - started) * 1000
            result = response.json()
            sent.append(
                timestamp=timestamp,
                camera=camera,
                people=result["people"],
                inference_ms=result["inference_ms"],
                response_ms=response_ms,
                bytes_sent=len(jpeg),
            )
            processed += 1
            log.info("%s: %d people (%.0f ms round trip)", camera, result["people"], response_ms)
    finally:
        if processed:
            plot_counts(sent.path, output / "people-over-time.png")
            log.info("Sent %d frames; log and graph in %s", processed, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="VM address, e.g. http://20.1.2.3")
    parser.add_argument("--camera", choices=CAMERAS, default="gaggenau")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames")
    parser.add_argument("--frames", type=int, help="stop after N frames (default: until Ctrl+C)")
    parser.add_argument("--output", type=Path, help="log folder (default: runs/sent-<camera>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        send(
            capture(CAMERAS[args.camera], args.interval, args.frames),
            args.server.rstrip("/"),
            os.environ["SMART_LIGHTING_TOKEN"],
            args.camera,
            args.output or Path("runs") / f"sent-{args.camera}",
        )
    except KeyboardInterrupt:
        log.info("Stopped by user")


if __name__ == "__main__":
    main()
